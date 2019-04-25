import pytest
# import tensorflow as tf
# import tensorflow.test.TestCase
# from tensorflow.contrib.eager.python import tfe


class TestTensorflow(object):
    def test_one(self):
        x = "this"
        assert 'h' in x

    # @pytest.fixture(scope='function', autouse=True)
    # def eager(self, request):
    #     tfe.enable_eager_execution()
    #
    # def test_func(self):
    #     x = tf.Variable(3, name="x")
    #     y = tf.Variable(4, name="y")
    #     f = x*x*y + y + 2
    #     with tf.Session() as sess:
    #         x.initializer.run()
    #         y.initializer.run()
    #         result = f.eval()
    #         assert result == 42

    # def test_spam(self):
    #     assert True
    #
    # def test_eggs(self):
    #     assert True
    #
    # def test_bacon(self):
    #     assert True