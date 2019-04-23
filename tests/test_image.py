import os, pytest

CONTENT = u"content"

class TestImage(object):
    def test_one(self):
        x = "this"
        assert 'h' in x

    @pytest.mark.xfail()
    def test_two(self):
        x = "hello"
        assert hasattr(x, 'format')

    # content of test_tmpdir.py
    def test_needsfiles(tmpdir):
        print(tmpdir)
        assert 1

    @pytest.mark.xfail()
    def test_create_file(tmp_path):
        d = tmp_path / "sub"
        d.mkdir()
        p = d / "hello.txt"
        p.write_text(CONTENT)
        assert p.read_text() == CONTENT
        assert len(list(tmp_path.iterdir())) == 1
        assert 0

    @pytest.fixture(scope="session")
    def image_file(tmpdir_factory):
        img = compute_expensive_image()
        fn = tmpdir_factory.mktemp("data").join("img.png")
        img.save(str(fn))
        return fn

    # contents of test_image.py
    def test_histogram(image_file):
        img = load_image(image_file)

    @pytest.fixture(scope="session")
    def s1(self):
        print ("session fixture")
        pass

    @pytest.fixture(scope="module")
    def m1(self):
        print ("module fixture")
        pass

    @pytest.fixture
    def f1(tmpdir):
        print ("tempdirn fixture")
        pass

    @pytest.fixture
    def f2(self):
        print ("func fixture")
        pass

    def test_foo(self, f1, m1, f2, s1):
        assert 1