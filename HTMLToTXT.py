import email
import pdfkit
import pdfplumber
import re

def mhtmlToPDF(mhtml_path: str, pdf_path: str):
    try:
        with open(mhtml_path, "r", encoding="utf-8") as f:
            msg = email.message_from_file(f)

        for part in msg.walk():
            if part.get_content_type() == "text/html":
                html_content = part.get_payload(decode=True).decode()
                try:
                    pdfkit.from_string(html_content, pdf_path)
                    print(f"PDF 已生成: {pdf_path}")
                except Exception as e:
                    print(f"PDF 转换时出现异常，但继续执行: {e}")
                break  # 找到第一个 HTML 内容即可
    except Exception as e:
        print(f"MHTML 读取或解析异常: {e}")


def extract_text_clean(pdf_path):
    """
    从 PDF 提取文本，并清理掉特殊字符，只保留中文、英文、数字、空格和常用标点
    """
    all_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                all_text += page_text + "\n"

    clean_text = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9\s\.,;!?()\-]", "", all_text)
    return clean_text

def save_to_txt(text, txt_path):
    """
    将文本保存到指定路径
    """
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

def main():
    mhtml_path = "万字长文讲透LLM核心：Transformer架构原理解析.mhtml"
    pdf_path = "example.pdf"
    txt_path = "output.txt"

    # 读取 MHTML 文件并转换为 PDF
    mhtmlToPDF(mhtml_path, pdf_path)

    # 从 PDF 提取文本并保存
    try:
        text = extract_text_clean(pdf_path)
        save_to_txt(text, txt_path)
        print(f"文本已成功保存到 {txt_path}")
    except Exception as e:
        print(f"PDF 文本提取或保存异常: {e}")




if __name__ == "__main__":
    main()