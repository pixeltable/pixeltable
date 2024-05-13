import pathlib
import tempfile

from pixeltable.utils.transactional_directory import transactional_directory


class MyException(Exception):
    pass

class TestTransactionalDirectory:
    def test_success(self) -> None:
        test_dir = pathlib.Path(tempfile.mkdtemp())
        assert test_dir.exists()
        final = test_dir / "test_success"
        assert not final.exists()
        with transactional_directory(final) as folder:
            assert folder.exists()
            (folder / "subfolder1").mkdir()
            with (folder / "test.txt").open("w") as f:
                f.write("test")

        assert final.exists()
        assert (final / "subfolder1").is_dir()
        assert (final / "test.txt").read_text() == "test"

    def test_failure(self) -> None:
        test_dir = pathlib.Path(tempfile.mkdtemp())
        assert test_dir.exists()
        final = test_dir / "test_failure"
        assert not final.exists()

        try:
            with transactional_directory(final) as folder:
                assert folder.exists()
                (folder / "subfolder1").mkdir()
                with (folder / "test.txt").open("w") as f:
                    f.write("test")
                raise MyException()
        except MyException:
            pass

        assert not final.exists()
