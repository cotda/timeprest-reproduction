# Hướng dẫn sử dụng Template Slide Beamer - Khoa HTTT - UIT

Chào các bạn sinh viên, đây là tài liệu hướng dẫn sử dụng và tùy biến mẫu slide báo cáo (Beamer LaTeX) được thiết kế theo phong cách chuyên nghiệp, hiện đại, thích hợp cho báo cáo môn học, báo cáo tiến độ nghiên cứu khoa học, khóa luận tốt nghiệp, v.v.

Mẫu slide này đã được cấu hình sẵn với các quy chuẩn về giao diện, chống tràn khung chữ, hỗ trợ ảnh nền tiêu đề mờ, tự động hiển thị logo đôi (nếu có) và tích hợp các hộp thông tin trực quan.

---

## 1. Cấu trúc thư mục tối thiểu
Để slide biên dịch hoạt động hoàn hảo và không bị lỗi hình ảnh, các bạn nên tổ chức các file trong thư mục như sau:
```text
├── template.tex              # File nguồn LaTeX chứa nội dung slide mẫu
├── template.pdf              # File kết quả biên dịch thử nghiệm
├── title_bg.png              # Hình nền mờ cho trang tiêu đề chính (nếu có)
├── logo_nhom_nc.png          # Logo nhóm nghiên cứu/Lab (nếu có)
├── logo_nhom_thuchien.png    # Logo nhóm thực hiện/CLB (nếu có)
└── images/                   # Thư mục chứa các hình ảnh minh họa trong slide
```

---

## 2. Các khu vực tùy chỉnh chính trong file `.tex`

Mọi tùy chỉnh về thương hiệu, thông tin cá nhân và màu sắc đều được quy tụ về **phần đầu của file `.tex`** (từ dòng 21 đến dòng 45). Các bạn **không cần** sửa đổi phần cấu hình hệ thống bên dưới.

### A. Tùy chỉnh màu sắc (Theme Colors)
Mặc định template sử dụng tông màu **Nâu đất & Vàng Gold (Coffee/Gold)** sang trọng. Các bạn có thể thay đổi mã màu RGB để tạo ra phong cách riêng:

```latex
\definecolor{uitblue}{RGB}{108,79,59}       % Màu chủ đạo (Nâu đậm của header/bìa)
\definecolor{uitdark}{RGB}{74,53,37}        % Màu phụ (Nâu tối ở dải bìa/footer)
\definecolor{accent}{RGB}{197,140,50}       % Màu nhấn mạnh (Vàng Gold)
\definecolor{accentlt}{RGB}{222,178,110}    % Màu nhấn phụ (Vàng cát)
\definecolor{greenteal}{RGB}{100,120,80}    % Màu ví dụ/thực hành (Sage Green)
\definecolor{lightgray}{RGB}{248,245,240}   % Màu nền ấm của slide (Cream/Ivory)
\definecolor{textgray}{RGB}{60,55,50}       % Màu chữ thân (Charcoal)
```
*Gợi ý:* Nếu muốn chuyển về tông **Xanh dương truyền thống của UIT**, hãy đổi `uitblue` thành `RGB{0, 90, 160}` và `uitdark` thành `RGB{0, 60, 110}`.

### B. Tùy chỉnh Logo & Hình nền trang bìa
Template hỗ trợ cơ chế kiểm tra sự tồn tại của tệp thông minh (`\IfFileExists`). Nếu tệp logo không tồn tại trong thư mục, hệ thống sẽ tự ẩn đi mà không gây lỗi biên dịch:

```latex
\newcommand{\titlebgimage}{title_bg.png}     % Tệp ảnh nền mờ trang bìa (khuyên dùng định dạng chìm/mờ)
\newcommand{\logonhomNC}{logo_nhom_nc.png}   % Tệp logo Nhóm nghiên cứu/Lab (bên phải ngoài cùng)
\newcommand{\logonhomthuchien}{logo_nhom_thuchien.png} % Tệp logo Nhóm nhỏ/Câu lạc bộ (bên cạnh)
\newcommand{\contactemail}{ftisu@uit.edu.vn} % Email liên lạc hiển thị ở dải thông tin trang bìa
```

### C. Thay đổi thông tin báo cáo
```latex
\title[Tiêu đề rút gọn]{Tiêu đề đầy đủ của bài báo cáo hoặc đề tài nghiên cứu}
\subtitle{Tên môn học hoặc Tên đề tài báo cáo khoa học}
\author{Nhóm thực hiện: Nguyễn Văn A - Lớp HTTT202X}
\date{Ngày báo cáo: \today}
```

---

## 3. Các thành phần giao diện hữu ích được định nghĩa sẵn

### A. Hộp nội dung nổi bật (Custom Boxes)
Template cung cấp 3 loại hộp nội dung với màu sắc hài hòa để phân loại thông tin:
*   **Hộp mặc định (`uitbox`):** Màu chủ đạo nâu đất, thích hợp cho định nghĩa, giới thiệu vấn đề.
    ```latex
    \begin{uitbox}{\faLightbulb\; Đặt vấn đề}
      Nội dung khái quát về bài toán hoặc đề tài nghiên cứu.
    \end{uitbox}
    ```
*   **Hộp nhấn mạnh (`accentbox`):** Màu vàng cát, thích hợp làm nổi bật công thức, thuật toán quan trọng.
    ```latex
    \begin{accentbox}{\faBalanceScale\; Hàm mục tiêu tối ưu}
      $$f(x) = \dots$$
    \end{accentbox}
    ```
*   **Hộp thực hành/Ví dụ (`greenbox`):** Màu xanh lá, thích hợp ghi chú ví dụ, kết quả thực nghiệm.
    ```latex
    \begin{greenbox}{\faPlayCircle\; Sơ đồ minh họa}
      % Nội dung vẽ hoặc chèn hình ảnh
    \end{greenbox}
    ```

### B. Lệnh tô màu từ khóa nhanh
*   Sử dụng `\highlight{từ khóa}` để tô đậm màu vàng nhấn.
*   Sử dụng `\kw{từ khóa}` để tô đậm màu nâu chủ đạo.

### C. Chia cột linh hoạt (Layout 2 cột)
Khi cần chèn sơ đồ hoặc bảng dữ liệu một bên, chữ mô tả một bên, hãy sử dụng môi trường `columns`:
```latex
\begin{columns}[T]
  \begin{column}{0.50\textwidth}
    % Cột bên trái: Chiếm 50% độ rộng slide
    \begin{uitbox}{Mô tả}
      Nội dung phân tích cột trái.
    \end{uitbox}
  \end{column}
  
  \begin{column}{0.45\textwidth}
    % Cột bên phải: Chiếm 45% độ rộng slide
    \centering
    \includegraphics[width=\textwidth]{images/so_do.png}
  \end{column}
\end{columns}
```

---

## 4. Lưu ý quan trọng tránh lỗi khi làm Slide
1.  **Tránh Overlap Công thức:** Các công thức toán có kích thước lớn hoặc phân số nhiều tầng (như tích phân, tổng chuỗi phức tạp) nên được trình bày ở chế độ hiển thị toàn màn hình (sử dụng `$$` hoặc `\[ \]`) trên một slide riêng biệt thay vì chèn vào các hộp thông tin dạng 2 cột hẹp để tránh tràn biên hoặc đè chữ.
2.  **Đường kẻ ngăn chương:** Mỗi khi sử dụng lệnh `\section{...}`, hệ thống sẽ tự động sinh ra một trang bìa ngăn chương với màu nền chủ đạo và đường kẻ vàng sang trọng nằm ở chính giữa, đảm bảo thông tin số chương và tên chương không bị đè lên nhau.
3.  **Tối ưu hóa hình ảnh:** Khi chèn hình ảnh minh họa, hãy luôn chỉ định chiều rộng hoặc chiều cao tương đối (ví dụ: `width=0.8\textwidth` hoặc `height=0.6\textheight`) để đảm bảo ảnh không vượt quá tỷ lệ màn hình 16:9 của slide.

---

## 5. Hướng dẫn biên dịch (Compile)
Để có kết quả hiển thị tốt nhất, khuyến nghị biên dịch bằng **pdfLaTeX** hoặc **XeLaTeX** (nếu dùng font chữ hệ thống khác):
*   Nếu dùng các IDE offline (như TeXstudio, Texmaker): Chọn chế độ biên dịch là `pdfLaTeX` $\rightarrow$ `View PDF`.
*   Nếu dùng công cụ trực tuyến **Overleaf**: Hãy tải tất cả các tệp logo, hình nền lên cùng một dự án và cấu hình Compiler là `pdfLaTeX`.

Chúc các bạn có những buổi báo cáo thành công tốt đẹp!
