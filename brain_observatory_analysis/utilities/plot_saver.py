import os
import matplotlib.pyplot as plt
from pathlib import Path

# make valid file extensions all possible image saving extensions
VALID_EXTENSIONS = ['.png', '.pdf', '.svg', '.eps', '.ps', '.raw', '.rgba', '.jpg', '.jpeg', '.tif', '.tiff']
RASTER_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.raw', '.rgba']
VECTOR_EXTENSIONS = ['.pdf', '.svg', '.eps', '.ps']
from typing import Optional, Union


class PlotSaver:
    def __init__(self, 
                 folder: Union[str, Path] = '',
                 filename: str = 'figure.png',
                 sub_folder: Optional[str] = None,
                 overwrite=True,
                 vector_and_raster=True):
        self.folder = folder
        self.filename = filename
        self.sub_folder = sub_folder
        self.overwrite = overwrite
        self.vector_and_raster = vector_and_raster  # not implmented

        self.filename = self._check_file_extension(self.filename)

    def save(self, fig):

        if self.sub_folder is not None:
            self.folder = Path(self.folder) / self.sub_folder
            self._make_final_path()
            filepath = Path(self.folder) / self.filename
        else:
            self._make_final_path()
            filepath = Path(self.folder) / self.filename

        if not self.overwrite and filepath.exists():
            raise FileExistsError(f'File already exists: {filepath}')

        fig.savefig(filepath, dpi=300, bbox_inches='tight')

    def _make_final_path(self):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def _check_file_extension(self, filename: str):
        print(self.filename)
        if not any(filename.endswith(ext) for ext in VALID_EXTENSIONS):
            filename = filename + '.png'
        return filename
