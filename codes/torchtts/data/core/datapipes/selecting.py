from typing import Callable, TypeVar, Iterator, Optional, Tuple, Dict

from torchtts.data.core.datapipes.iter_datapipe import functional_datapipe
from torchtts.data.core.datapipes.iter_datapipe import IterDataPipe
from torchtts.data.core.datapipes.callable import MapIterDataPipe

T_co = TypeVar("T_co", covariant=True)


@functional_datapipe("filter")
class FilterIterDataPipe(MapIterDataPipe[T_co]):
    r""":class:`FilterIterDataPipe`.
    Iterable DataPipe to filter elements from datapipe according to filter_fn.
    args:
        datapipe: Iterable DataPipe being filtered
        filter_fn: Customized function mapping an element to a boolean.
        fn_args: Positional arguments for `filter_fn`
        fn_kwargs: Keyword arguments for `filter_fn`
    """

    def __init__(
        self,
        datapipe: IterDataPipe[T_co],
        filter_fn: Callable[..., bool],
        fn_args: Optional[Tuple] = None,
        fn_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__(datapipe, fn=filter_fn, fn_args=fn_args, fn_kwargs=fn_kwargs)

    def __iter__(self) -> Iterator[T_co]:
        res: bool
        for data in self.datapipe:
            res = self.fn(data, *self.args, **self.kwargs)
            if not isinstance(res, bool):
                raise ValueError("Boolean output is required for " "`filter_fn` of FilterIterDataPipe")
            if res:
                yield data

    def __len__(self):
        raise NotImplementedError
