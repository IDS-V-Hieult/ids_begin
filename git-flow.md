■Quy trình commit code cho 2 branch
blog_writting
main

■Đầu tiên, kiểm tra trạng thái các file đã thay đổi ở nhánh "blog_writting":

git status

■Thêm các file đã thay đổi vào staging area:
git add .           # Thêm tất cả file đã thay đổi

Hoặc

git add <tên_file>  # Thêm từng file cụ thể

■Commit các thay đổi với message mô tả rõ ràng:
git commit -m "feat: add blog writing feature"

(Lưu ý: Nên tuân thủ conventional commits để message dễ đọc và quản lý: feat/fix/docs/style/refactor/test/chore)

Đảm bảo branch local của bạn đã cập nhật với branch main mới nhất:

■Pull các thay đổi mới nhất từ remote về "blog_writting"
git pull origin blog_writting

■Push code lên remote repository:
git push origin blog_writting

■Nếu đây là lần đầu push branch này lên remote, bạn có thể cần:
git push --set-upstream origin blog_writting
hoặc

git push -u origin blog_writting

■Một số lưu ý quan trọng:

Luôn kiểm tra bạn đang ở đúng branch trước khi làm việc:
git branch
Commit message nên rõ ràng, mô tả được những gì đã thay đổi

Sau khi push thành công, bạn có thể tạo Pull Request (PR) từ branch "blog_writting"
vào main trên giao diện GitHub/GitLab để team review code.
Sau khi review code OK, thì tiến hành merge vào "main" trên giao diện github

■Giải thích về lệnh
- **git pull origin blog_writting**

git pull: Kéo (tải về) và cập nhật code từ kho lưu trữ từ xa (remote repository) về máy tính của bạn

origin: Tên mặc định của kho lưu trữ từ xa, thường là repository gốc trên GitHub/GitLab/Bitbucket

blog_writting: Tên nhánh (branch) mà bạn muốn kéo về

Nói cách khác, lệnh này sẽ:

Kết nối đến repository từ xa có tên là "origin"

Tìm nhánh có tên "blog_writting" trên repository đó

Tải về tất cả các thay đổi mới từ nhánh "blog_writting"

Tự động merge (hợp nhất) những thay đổi đó vào nhánh hiện tại trên máy tính của bạn


■Giải thích về lệnh
- **git push origin blog_writting**

git push: Đẩy (tải lên) code từ máy tính của bạn lên kho lưu trữ từ xa (remote repository)

origin: Tên mặc định của kho lưu trữ từ xa, thường là repository gốc trên GitHub/GitLab/Bitbucket

blog_writting: Tên nhánh (branch) mà bạn muốn đẩy lên

Nói cách khác, lệnh này sẽ:

Kết nối đến repository từ xa có tên là "origin"

Tìm nhánh có tên "blog_writting" trên repository đó

Tải lên tất cả các thay đổi từ nhánh "blog_writting" trên máy tính của bạn lên repository từ xa


■Khi muốn merge nội dung cập nhật từ nhánh "blog_writting" vào nhánh "main"
# Rebase với main để có nội dung mới nhất từ "main"
git rebase origin/main

# Nếu có conflict:
# 1. Giải quyết từng file conflict
# 2. Add và commit các file đã resolve
git commit -am "message hợp lý"
# 3. Tiếp tục rebase
git rebase --continue

# 4. Nếu muốn hủy rebase
git rebase --abort


# 5. Sau đó push lại nội dung vào nhánh "blog_writting" trên remote
git push origin blog_writting


# Tạo Pull Request từ "blog_writting" --> "main" sau đó merge vào "main" bằng tool trên Github


■So sánh git merge và git rebase
# Git Merge - "Giữ nguyên lịch sử"
```python
A---B---C (main)
     \
      D---E (feature)
       \     \
        \     \
         \     \
          ------M (sau khi merge)
```
Đặc điểm:

Tạo ra một commit mới (M) gọi là "merge commit"

Giữ lại toàn bộ lịch sử của cả 2 nhánh

Có thể thấy rõ code đến từ nhánh nào

Lịch sử commit hiển thị dạng "song song" với các nhánh rẽ

# Git Rebase - "Viết lại lịch sử"
```python
A---B---C (main)
         \
          D'---E' (feature sau khi rebase)
```

Đặc điểm:

Di chuyển toàn bộ các commit từ nhánh feature (D, E) lên đầu nhánh main

Tạo ra các commit mới (D', E') có nội dung giống commit cũ nhưng có hash mới

Lịch sử trở thành một đường thẳng, không còn thấy được điểm rẽ nhánh ban đầu

Trông như thể các commit của feature được viết tiếp theo main một cách tuần tự

#main blog_writting