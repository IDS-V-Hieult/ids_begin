■Quy trình commit code cho 2 branch
blog_writting
main

■Đầu tiên, kiểm tra trạng thái các file đã thay đổi:

git status

■Thêm các file đã thay đổi vào staging area:
git add .           # Thêm tất cả file đã thay đổi

Hoặc

git add <tên_file>  # Thêm từng file cụ thể

■Commit các thay đổi với message mô tả rõ ràng:
git commit -m "feat: add blog writing feature"

(Lưu ý: Nên tuân thủ conventional commits để message dễ đọc và quản lý: feat/fix/docs/style/refactor/test/chore)

Đảm bảo branch local của bạn đã cập nhật với branch main mới nhất:

■Chuyển về branch main
git checkout main

■Pull các thay đổi mới nhất từ remote main
git pull origin main

■Quay lại branch blog_writting
git checkout blog_writting

■Rebase với main để có code mới nhất và tránh conflict
git rebase main

■Pull các thay đổi mới nhất từ origin xuống blog_writting
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
Nên rebase với main trước khi push để tránh conflicts
Nếu có conflict khi rebase:
■Giải quyết conflicts trong code
# Sau đó
git add .
git rebase --continue

■Nếu bạn cần hủy rebase:
git rebase --abort


Sau khi push thành công, bạn có thể tạo Pull Request (PR) từ branch "blog_writting"
vào main trên giao diện GitHub/GitLab để team review code.