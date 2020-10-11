from models.steps.losses import ConfigurableLoss, GANLoss, extract_params_from_state, get_basic_criterion_for_name
from models.layers.resample2d_package.resample2d import Resample2d
from models.steps.recurrent import RecurrentController
from models.steps.injectors import Injector
import torch
import torch.nn.functional as F
import os
import os.path as osp
import torchvision
import torch.distributed as dist

def create_teco_loss(opt, env):
    type = opt['type']
    if type == 'teco_gan':
        return TecoGanLoss(opt, env)
    elif type == "teco_pingpong":
        return PingPongLoss(opt, env)
    return None

def create_teco_injector(opt, env):
    type = opt['type']
    if type == 'teco_recurrent_generated_sequence_injector':
        return RecurrentImageGeneratorSequenceInjector(opt, env)
    elif type == 'teco_flow_adjustment':
        return FlowAdjustment(opt, env)
    return None

def create_teco_discriminator_sextuplet(input_list, lr_imgs, scale, index, flow_gen, resampler, margin):
    triplet = input_list[:, index:index+3]
    # Flow is interpreted from the LR images so that the generator cannot learn to manipulate it.
    with torch.no_grad():
        first_flow = flow_gen(torch.stack([triplet[:,1], triplet[:,0]], dim=2).float())
        #first_flow = F.interpolate(first_flow, scale_factor=scale, mode='bicubic')
        last_flow = flow_gen(torch.stack([triplet[:,1], triplet[:,2]], dim=2).float())
        #last_flow = F.interpolate(last_flow, scale_factor=scale, mode='bicubic')
    flow_triplet = [resampler(triplet[:,0].float(), first_flow.float()),
                    triplet[:,1],
                    resampler(triplet[:,2].float(), last_flow.float())]
    flow_triplet = torch.stack(flow_triplet, dim=1)
    combined = torch.cat([triplet, flow_triplet], dim=1)
    b, f, c, h, w = combined.shape
    combined = combined.view(b, 3*6, h, w)  # 3*6 is essentially an assertion here.
    # Apply margin
    return combined[:, :, margin:-margin, margin:-margin]


def extract_inputs_index(inputs, i):
    res = []
    for input in inputs:
        if isinstance(input, torch.Tensor):
            res.append(input[:, i])
        else:
            res.append(input)
    return res

# Uses a generator to synthesize a sequence of images from [in] and injects the results into a list [out]
# Images are fed in sequentially forward and back, resulting in len([out])=2*len([in])-1 (last element is not repeated).
# All computation is done with torch.no_grad().
class RecurrentImageGeneratorSequenceInjector(Injector):
    def __init__(self, opt, env):
        super(RecurrentImageGeneratorSequenceInjector, self).__init__(opt, env)
        self.flow = opt['flow_network']
        self.input_lq_index = opt['input_lq_index'] if 'input_lq_index' in opt.keys() else 0
        self.output_hq_index = opt['output_hq_index'] if 'output_hq_index' in opt.keys() else 0
        self.recurrent_index = opt['recurrent_index']
        self.scale = opt['scale']
        self.resample = Resample2d()
        self.first_inputs = opt['first_inputs'] if 'first_inputs' in opt.keys() else opt['in']  # Use this to specify inputs that will be used in the first teco iteration, the rest will use 'in'.
        self.do_backwards = opt['do_backwards'] if 'do_backwards' in opt.keys() else True

    def forward(self, state):
        gen = self.env['generators'][self.opt['generator']]
        flow = self.env['generators'][self.flow]
        results = []
        first_inputs = extract_params_from_state(self.first_inputs, state)
        inputs = extract_params_from_state(self.input, state)
        if not isinstance(inputs, list):
            inputs = [inputs]

        # Go forward in the sequence first.
        first_step = True
        b, f, c, h, w = inputs[self.input_lq_index].shape
        debug_index = 0
        for i in range(f):
            if first_step:
                input = extract_inputs_index(first_inputs, i)
                recurrent_input = torch.zeros_like(input[self.recurrent_index])
                first_step = False
            else:
                input = extract_inputs_index(inputs, i)
                with torch.no_grad():
                    reduced_recurrent = F.interpolate(recurrent_input, scale_factor=1/self.scale, mode='bicubic')
                    flow_input = torch.stack([input[self.input_lq_index], reduced_recurrent], dim=2)
                    flowfield = F.interpolate(flow(flow_input), scale_factor=self.scale, mode='bicubic')
                    # Resample does not work in FP16.
                    recurrent_input = self.resample(reduced_recurrent.float(), flowfield.float())
            input[self.recurrent_index] = recurrent_input
            if self.env['step'] % 50 == 0:
                self.produce_teco_visual_debugs(input[self.input_lq_index], debug_index)
                debug_index += 1
            gen_out = gen(*input)
            if isinstance(gen_out, torch.Tensor):
                gen_out = [gen_out]
            recurrent_input = gen_out[self.output_hq_index]
            results.append(recurrent_input)

        # Now go backwards, skipping the last element (it's already stored in recurrent_input)
        if self.do_backwards:
            it = reversed(range(f - 1))
            for i in it:
                input = extract_inputs_index(inputs, i)
                with torch.no_grad():
                    reduced_recurrent = F.interpolate(recurrent_input, scale_factor=1 / self.scale, mode='bicubic')
                    flow_input = torch.stack([input[self.input_lq_index], reduced_recurrent], dim=2)
                    flowfield = F.interpolate(flow(flow_input), scale_factor=self.scale, mode='bicubic')
                    recurrent_input = self.resample(reduced_recurrent.float(), flowfield.float())
                input[self.recurrent_index
                ] = recurrent_input
                if self.env['step'] % 50 == 0:
                    self.produce_teco_visual_debugs(input[self.input_lq_index], debug_index)
                    debug_index += 1
                gen_out = gen(*input)
                if isinstance(gen_out, torch.Tensor):
                    gen_out = [gen_out]
                recurrent_input = gen_out[self.output_hq_index]
                results.append(recurrent_input)

        return {self.output: results}

    def produce_teco_visual_debugs(self, gen_input, it):
        if self.env['rank'] > 0:
            return
        base_path = osp.join(self.env['base_path'], "..", "visual_dbg", "teco_geninput", str(self.env['step']))
        os.makedirs(base_path, exist_ok=True)
        torchvision.utils.save_image(gen_input[:, :3], osp.join(base_path, "%s_img.png" % (it,)))
        torchvision.utils.save_image(gen_input[:, 3:], osp.join(base_path, "%s_recurrent.png" % (it,)))


class FlowAdjustment(Injector):
    def __init__(self, opt, env):
        super(FlowAdjustment, self).__init__(opt, env)
        self.resample = Resample2d()
        self.flow = opt['flow_network']
        self.flow_target = opt['flow_target']
        self.flowed = opt['flowed']

    def forward(self, state):
        flow = self.env['generators'][self.flow]
        flow_target = state[self.flow_target]
        flowed = state[self.flowed]
        flow_input = torch.stack([flow_target, flowed], dim=2)
        flowfield = flow(flow_input)
        return {self.output: self.resample(flowed.float(), flowfield.float())}


# This is the temporal discriminator loss from TecoGAN.
#
# It has a strict contract for 'real' and 'fake' inputs:
#   'real' - Must be a list of arbitrary images (len>3) drawn from the dataset
#   'fake' - The output of the RecurrentImageGeneratorSequenceInjector for the same set of images.
#
# This loss does the following:
# 1) Picks an image triplet, starting with the first '3' elements in 'real' and 'fake'.
# 2) Uses the image flow generator (specified with 'image_flow_generator') to create detached flow fields for the first and last images in the above sequence.
# 3) Warps the first and last images according to the flow field.
# 4) Composes the three base image and the 2 warped images and middle image into a tensor concatenated at the filter dimension for both real and fake, resulting in a bx18xhxw shape tensor.
# 5) Feeds the catted real and fake image sets into the discriminator, computes a loss, and backward().
# 6) Repeat from (1) until all triplets from the real sequence have been exhausted.
class TecoGanLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(TecoGanLoss, self).__init__(opt, env)
        self.criterion = GANLoss(opt['gan_type'], 1.0, 0.0).to(env['device'])
        # TecoGAN parameters
        self.scale = opt['scale']
        self.lr_inputs = opt['lr_inputs']
        self.image_flow_generator = opt['image_flow_generator']
        self.resampler = Resample2d()
        self.for_generator = opt['for_generator']
        self.min_loss = opt['min_loss'] if 'min_loss' in opt.keys() else 0
        self.margin = opt['margin']  # Per the tecogan paper, the GAN loss only pays attention to an inner part of the image with the margin removed, to get rid of artifacts resulting from flow errors.

    def forward(self, _, state):
        net = self.env['discriminators'][self.opt['discriminator']]
        flow_gen = self.env['generators'][self.image_flow_generator]
        real = state[self.opt['real']]
        fake = torch.stack(state[self.opt['fake']], dim=1)
        sequence_len = real.shape[1]
        lr = state[self.opt['lr_inputs']]
        l_total = 0
        for i in range(sequence_len - 2):
            real_sext = create_teco_discriminator_sextuplet(real, lr, self.scale, i, flow_gen, self.resampler, self.margin)
            fake_sext = create_teco_discriminator_sextuplet(fake, lr, self.scale, i, flow_gen, self.resampler, self.margin)
            d_fake = net(fake_sext)
            d_real = net(real_sext)
            self.metrics.append(("d_fake", torch.mean(d_fake)))
            self.metrics.append(("d_real", torch.mean(d_real)))

            if self.for_generator and self.env['step'] % 50 == 0:
                self.produce_teco_visual_debugs(fake_sext, 'fake', i)
                self.produce_teco_visual_debugs(real_sext, 'real', i)

            if self.opt['gan_type'] in ['gan', 'pixgan']:
                l_fake = self.criterion(d_fake, self.for_generator)
                if not self.for_generator:
                    l_real = self.criterion(d_real, True)
                else:
                    l_real = 0
                l_step = l_fake + l_real
            elif self.opt['gan_type'] == 'ragan':
                d_fake_diff = d_fake - torch.mean(d_real)
                self.metrics.append(("d_fake_diff", torch.mean(d_fake_diff)))
                l_step = (self.criterion(d_real - torch.mean(d_fake), not self.for_generator) +
                           self.criterion(d_fake_diff, self.for_generator))
            else:
                raise NotImplementedError
            if l_step > self.min_loss:
                l_total += l_step

        return l_total

    def produce_teco_visual_debugs(self, sext, lbl, it):
        if self.env['rank'] > 0:
            return
        base_path = osp.join(self.env['base_path'], "..", "visual_dbg", "teco_sext", str(self.env['step']), lbl)
        os.makedirs(base_path, exist_ok=True)
        lbls = ['img_a', 'img_b', 'img_c', 'flow_a', 'flow_b', 'flow_c']
        for i in range(6):
            torchvision.utils.save_image(sext[:, i*3:(i+1)*3, :, :], osp.join(base_path, "%s_%s.png" % (it, lbls[i])))


# This loss doesn't have a real entry - only fakes are used.
class PingPongLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super(PingPongLoss, self).__init__(opt, env)
        self.opt = opt
        self.criterion = get_basic_criterion_for_name(opt['criterion'], env['device'])

    def forward(self, _, state):
        fake = state[self.opt['fake']]
        l_total = 0
        for i in range((len(fake) - 1) // 2):
            early = fake[i]
            late = fake[-i]
            l_total += self.criterion(early, late)

        if self.env['step'] % 50 == 0:
            self.produce_teco_visual_debugs(fake)

        return l_total

    def produce_teco_visual_debugs(self, imglist):
        if self.env['rank'] > 0:
            return
        base_path = osp.join(self.env['base_path'], "..", "visual_dbg", "teco_pingpong", str(self.env['step']))
        os.makedirs(base_path, exist_ok=True)
        assert isinstance(imglist, list)
        for i, img in enumerate(imglist):
            torchvision.utils.save_image(img, osp.join(base_path, "%s.png" % (i, )))
