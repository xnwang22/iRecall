# iRecall
test image recognition with tensor flow
1) to clone: git clone https://github.com/xnwang22/iRecall.git
2) to install dependency: pip install -r requirements_dev.txt
3) to run test: pytest tests/test_opencv.py

after fork
git remote add upstream https://github.com/xnwang22/iRecall.git
git remote set_url origin your_fork_url

git fetch origin
git checkout -b dev
git pull upstream dev
...make code change
git status
git commit -a -m "commit message"

git push origin HEAD:dev ** this will push your code to your fork

GoTO Browser, should see make Pull Request button...


