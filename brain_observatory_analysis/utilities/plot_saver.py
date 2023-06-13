import os
import matplotlib.pyplot as plt
import matplotlib as mpl
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


# another way of figure saving, including params settings to create illustrator editable PDFs
def save_figure(fig, figsize, save_dir, folder, fig_title, formats=['.png', '.pdf']):
    """
    Function to save a matplotlib figure to a specified directory and subfolder, 
    with filename as an argument that can be iterated over to deposit many plots in a single folder.
    if 'folder' does not exist, it will be created
    if '.pdf' is included in formats, PDF will be saved in illustrator editable format

    fig: matplotlib figure handle
    figsize: tuple with desired figsize
    save_dir: base directory in which to save figures
    folder: sub-directory within save_dir where figures should be saved
            if sub-directory folder does not exist in save_dir, it will be created
            Useful when iterating over many figures that should be saved to the same folder, 
            or over multiple types of plots saved in different folders in save_dir
    fig_title: filename of the saved figure
    formats: list options for PNG or PDF file formats

    """
    fig_dir = os.path.join(save_dir, folder)
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    filename = os.path.join(fig_dir, fig_title)
    mpl.rcParams['pdf.fonttype'] = 42
    fig.set_size_inches(figsize)
    for f in formats:
        fig.savefig(filename + f, transparent=True, orientation='landscape', bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor())
