from contextlib import contextmanager
import os
import warnings
import shutil
from  pathlib  import Path

@contextmanager
def transactional_folder(folder_path : Path, force=False):
    '''
        Example usage
        folder_path = "/path/to/folder"
        with transactional_folder(folder_path) as temp_folder:
            Perform file system operations using the temp_folder path inside the 'with' block
            NB. does not change the current working directory, only provides a place to accumulate changes
            You can use temp_folder to create, modify, or delete files or other folders

        If everything is successful, the changes are now committed via an atomic move operation.
        If an exception occurred, no changes have been applied and temp folder with . name is deleted (best effort)

        When folder already exists,
        `force` means we overwrite the existing folder when it exists, but still atomically.

     '''
    # TODO: convert to using pathlib internally

    folder_path = str(folder_path)
    if os.path.exists(folder_path):
        if not os.path.isdir(folder_path):
            raise Exception(f'{folder_path} exits and is not a folder')
        
        if not force:
            raise Exception(f'Folder {folder_path} already exists. use "force=True" to allow overwrite')
                
        warnings.warn(f'folder {folder_path} already exists. overwriting due to force=True')


    folder_path = folder_path.rstrip('/')
    prepath = os.path.dirname(folder_path)
    name = os.path.basename(folder_path)
    temp_folder = os.path.join(prepath, '.temp_' + name)
    
    try:
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder, ignore_errors=True)
        
        # Create a temporary folder to store the changes
        os.makedirs(temp_folder)

        # Passes temporary folder to the code inside the 'with' block
        yield Path(temp_folder)

        # If everything succeeds, `commit' the changes by moving the temporary folder
        os.replace(temp_folder, folder_path)
    except Exception as e:
        # If an exception occurred, try to clean up the temporary folder (not critical if this fails)
        raise e