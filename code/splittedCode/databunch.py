from torchvision import datasets
from torchvision import transforms as T
import os

import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union

from PIL import Image

import torch
import torch.utils.data as data

import numpy as np

import ctypes
import multiprocessing as mp

import torchaudio
import random

import librosa

import scipy.ndimage as nd


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
AUDIO_EXTENSIONS = ('.wav','.mp3')


class DataBunch():

    def __init__(self, train_dl, valid_dl, c=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.c = c
        
    @property
    def train_ds(self): 
        return self.train_dl.dataset
    
    @property
    def valid_ds(self):
        return self.valid.dl.dataset


class ClassSpecificImageFolder(datasets.DatasetFolder):
    def __init__(
            self,
            root,
            dropped_classes=[],
            transform = None,
            target_transform = None,
            loader = datasets.folder.default_loader,
            is_valid_file = None,
    ):
        self.dropped_classes = dropped_classes
        super(ClassSpecificImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                       transform=transform,
                                                       target_transform=target_transform,
                                                       is_valid_file=is_valid_file)
        self.imgs = self.samples

    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        classes = [c for c in classes if c not in self.dropped_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


class WholeDataset(data.Dataset):

    _repr_indent = 4

    def __init__(
        self,
        root: str,
        n_samples,
        n_channels,
        height,
        width,
        dropped_classes=[],
        loader = datasets.folder.default_loader,
        extensions: Optional[Tuple[str, ...]] = IMG_EXTENSIONS,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root
        self.dropped_classes = dropped_classes
        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.resize = T.Resize((height, width))
        self.height = height
        self.width = width
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

        shared_array_base = mp.Array(ctypes.c_ubyte, n_samples*n_channels*height*width)
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        self.shared_array = shared_array.reshape(n_samples, height, width, n_channels)
        self.use_cache = False

        self.classes, self.class_to_idx = self.find_classes(self.root)
        self.loader = loader
        self.extensions = extensions
        self.n_samples = n_samples
        self.samples = self.make_dataset(self.root, self.n_samples, self.class_to_idx, self.extensions, is_valid_file)

        self.targets = [s[1] for s in self.samples]

    def make_dataset(
        self,
        directory: str,
        n_samples,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (sample, class).

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")

        directory = os.path.expanduser(directory)

        if not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        #instances = np.empty((n_samples, 2), dtype=object)
        available_classes = set()
        index = 0
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        instances.append((path, class_index))
                        #instances.append((self.loader(path), class_index))
                        #instances[index] = self.loader(path), class_index
                        #index += 1
                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances

        return 

    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        classes = [c for c in classes if c not in self.dropped_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        if not self.use_cache:
            sample = self.loader(path)
            if sample.size != (self.width, self.height):
                sample = self.resize(sample)
            self.shared_array[index] = sample
        sample = self.shared_array[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def extra_repr(self) -> str:
        return ""


class StandardTransform:
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input: Any, target: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform, "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform, "Target transform: ")

        return "\n".join(body)


class ClassSpecificAudioFolder(datasets.DatasetFolder):
    def __init__(
            self,
            root,
            dropped_classes=[],
            transform = None,
            target_transform = None,
            loader = torchaudio.load,
            is_valid_file = None,
    ):
        self.dropped_classes = dropped_classes
        super(ClassSpecificAudioFolder, self).__init__(root, loader, AUDIO_EXTENSIONS if is_valid_file is None else None,
                                                       transform=transform,
                                                       target_transform=target_transform,
                                                       is_valid_file=is_valid_file)
        self.audios = self.samples

    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        classes = [c for c in classes if c not in self.dropped_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class AudioFolder(data.Dataset):

    _repr_indent = 4

    def __init__(
        self,
        root: str,
        dropped_classes=[],
        loader = torchaudio.load,
        extensions: Optional[Tuple[str, ...]] = AUDIO_EXTENSIONS,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root
        self.dropped_classes = dropped_classes
        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

        self.classes, self.class_to_idx = self.find_classes(self.root)
        self.loader = loader
        self.extensions = extensions
        self.samples = self.make_dataset(self.root, self.class_to_idx, self.extensions, is_valid_file)

        self.targets = [s[1] for s in self.samples]

    def make_dataset(
        self,
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (sample, class).

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")

        directory = os.path.expanduser(directory)

        if not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        instances.append((path, class_index))
                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances

    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        classes = [c for c in classes if c not in self.dropped_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample, sample_rate = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def extra_repr(self) -> str:
        return ""


class WholeAudioFolder(data.Dataset):

    _repr_indent = 4

    def __init__(
        self,
        root: str,
        dropped_classes=[],
        all_samples = False,
        num_samples_per_class = 2000,
        loader = torchaudio.load,
        extensions: Optional[Tuple[str, ...]] = AUDIO_EXTENSIONS,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root
        self.dropped_classes = dropped_classes
        self.all_samples = all_samples
        self.num_samples_per_class = num_samples_per_class
        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

        self.classes, self.class_to_idx = self.find_classes(self.root)
        self.loader = loader
        self.extensions = extensions
        self.samples, self.class_idx_to_indices = self.make_dataset(self.root, self.class_to_idx, self.extensions, is_valid_file)

        self.targets = [s[1] for s in self.samples]

    def make_dataset(
        self,
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (sample, class).

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")

        directory = os.path.expanduser(directory)

        if not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        class_idx_to_indices = dict()
        starting_idx_of_current_class = 0
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            previous_num_samples = 0
            target_num_samples = 0
            break_flag = False
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        if target_class not in available_classes:
                            available_classes.add(target_class)
                        offset_pos = self.sample_offset_pos(path, 88064)
                        previous_num_samples = target_num_samples
                        target_num_samples += len(offset_pos)
                        if target_num_samples <= self.num_samples_per_class or self.all_samples:
                            instances.extend([(path, class_index, x) for x in offset_pos])
                        else:
                            instances.extend([(path, class_index, x) for x in offset_pos[:self.num_samples_per_class-previous_num_samples]])
                            break_flag = True
                            break
                    if break_flag:
                        break
                if break_flag:
                    break
            if target_num_samples < self.num_samples_per_class and self.all_samples == False:
                num_duplicates = self.num_samples_per_class - target_num_samples
                instances.extend(random.choices(instances[-target_num_samples:],k=num_duplicates))
            class_idx_to_indices[class_index] = [starting_idx_of_current_class, len(instances)-1]
            starting_idx_of_current_class = len(instances)
                        
        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances, class_idx_to_indices

    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        classes = [c for c in classes if c not in self.dropped_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, offset = self.samples[index]
        sample, sample_rate = self.loader(path, frame_offset=offset, num_frames=88064)
        if sample.shape[1] < 88064:
            pad_len = 88064 - sample.shape[1]
            sample = torch.nn.functional.pad(sample, (0,pad_len), "constant", 0)
        if self.transform is not None:
            sample = self.transform(sample)
            sample = self.pre_filter(path, sample, offset, sample_rate, target, 3)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def extra_repr(self) -> str:
        return ""

    def sample_offset_pos(self, path, per_sample_frames):
        num_frames = torchaudio.info(path).num_frames
        return list(range(0,num_frames,per_sample_frames))

    def pre_filter(self, path ,sample, offset, sr, target, n_attempts):

        if n_attempts > 0:
            S_narrow = sample.reshape(128,345)

            #freq_pixels, time_pixels = S_narrow.shape

            #frequencies = librosa.fft_frequencies(sr=sr, n_fft=512)

            S_narrow = (S_narrow - S_narrow.min()) / (S_narrow.max() - S_narrow.min())
            
            print(S_narrow.shape)
            
            time_threshold = 3
            frequency_threshold = 3

            frequency_medians = torch.median(S_narrow, axis=1)[0] #  torch.median(s, axis)will return a tuple of (values, indices)?, no axis paramter(dim?)
            time_medians = torch.median(S_narrow, axis=0)[0]
            
            print("frequency_medians shape", frequency_medians.shape)
            print("time_medians shape", time_medians.shape)

            foreground = (S_narrow >= time_threshold * time_medians[None, :]) * \
                        (S_narrow >= frequency_threshold * frequency_medians[:, None])
            foreground_closed = nd.binary_closing(foreground)
            foreground_dilated = nd.binary_dilation(foreground_closed)

            foreground_median_filterd = nd.median_filter(foreground_dilated, size=2)
            print("foreground_median_filterd",foreground_median_filterd)
            print("nd.generate_binary_structure(2,2)",nd.generate_binary_structure(2,2))

            foreground_labeled, nb_labels = nd.label(foreground_median_filterd, structure=nd.generate_binary_structure(2,2))

            sizes = nd.sum(foreground_median_filterd, foreground_labeled, range(nb_labels + 1))
            foreground_sizes = sizes < 100
            remove_pixel = foreground_sizes[foreground_labeled]
            foreground_labeled[remove_pixel] = 0

            labels = np.unique(foreground_labeled)
            foreground_labeled = np.searchsorted(labels, foreground_labeled)

            #foreground_small_objects_removed = foreground_labeled > 0

            #print(foreground_small_objects_removed.shape)
            objects = nd.find_objects(foreground_labeled)
            print(objects)

            #bboxes = []
            if not objects: 
                print("bboxes are empty, find new sample! target: ",target," offset: ",offset," path: ",path)
                start_id, end_id = self.class_idx_to_indices[target]
                new_index = random.randint(start_id, end_id)
                path, new_target, new_offset = self.samples[new_index]
                if new_target != target:
                    print("FEHLER, new_target ist ungleich altes target -- new_target: ",new_target,"--- altes target: ", target)
                new_sample, new_sample_rate = self.loader(path, frame_offset=new_offset, num_frames=88064)
                if new_sample.shape[1] < 88064:
                    pad_len = 88064 - new_sample.shape[1]
                    new_sample = torch.nn.functional.pad(new_sample, (0,pad_len), "constant", 0)
                if self.transform is not None:
                    new_sample = self.transform(new_sample)
                    new_sample = self.pre_filter(new_sample, new_sample_rate, new_offset, new_target, n_attempts-1)
            else:
                return sample
        else:
            print("reached n_attemps, pre_filter unsuccesful, up n_attemps or tune parameter of pre_filter")
            return sample



        



