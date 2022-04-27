from typing import List
from os import path
from setuptools import find_packages
from setuptools import setup
import glob
from typing import List
import shutil


class Test:

    def get_model_zoo_configs(self) -> List[str]:
        """
        Return a list of configs to include in package for model zoo. Copy over these configs inside
        detectron2/model_zoo.
        """

        # Use absolute paths while symlinking.
        source_configs_dir = path.join(path.dirname(path.realpath(__file__)), "configs")
        destination = path.join(
            path.dirname(path.realpath(__file__)), "detectron2", "model_zoo", "configs"
        )
        print('src:', source_configs_dir)
        print('dest:', destination)
        # Symlink the config directory inside package to have a cleaner pip install.

        # Remove stale symlink/directory from a previous build.
        if path.exists(source_configs_dir):
            print('In first if')
            if path.islink(destination):
                path.unlink(destination)
            elif path.isdir(destination):
                shutil.rmtree(destination)

        if not path.exists(destination):
            try:
                print('in try block')
                path.symlink(source_configs_dir, destination)
            except OSError:
                # Fall back to copying if symlink fails: ex. on Windows.
                shutil.copytree(source_configs_dir, destination)

        '''config_paths = glob.glob(destination + '/*.yaml', recursive=True) + glob.glob(
            destination + '/*.py', recursive=True
        )'''
        config_paths = glob.glob("configs/**/*.yaml", recursive=True) + glob.glob(
            "configs/**/*.py", recursive=True
        )
        print(config_paths)
        return config_paths


if __name__ == "__main__":
    tst = Test()
    data = {"detectron2.model_zoo": tst.get_model_zoo_configs()}
    print(data)
