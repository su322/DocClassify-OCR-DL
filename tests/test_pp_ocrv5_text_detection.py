# PP-OCRv5 文本检测模块测试脚本
import os

# 辅助函数：生成测试图片
def ensure_test_image(path):
    if not os.path.exists(path):
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (400, 200), color=(255, 255, 255))
            d = ImageDraw.Draw(img)
            d.rectangle([50, 50, 350, 150], outline="black", width=3)
            d.text((60, 80), "Text Detection Test", fill=(0, 0, 0))
            img.save(path)
            print(f"已生成测试图片: {path}")
        except Exception as e:
            print(f"无法生成图片: {e}")

def run_test():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "test_images")
    os.makedirs(img_dir, exist_ok=True)
    image_path = os.path.join(img_dir, "general_ocr_001.png")

    ensure_test_image(image_path)

    try:
        from paddleocr import TextDetection
    except ImportError:
        print("错误: 无法导入 paddleocr.TextDetection。")
        return

    print("正在初始化 TextDetection 模型...")
    model = TextDetection()
    
    print(f"开始检测: {image_path}")
    output = model.predict(image_path)
    
    save_dir = os.path.join(current_dir, "output", "TextDetection_v5")
    os.makedirs(save_dir, exist_ok=True)

    for res in output: 
        res.print()
        res.save_to_img(save_path=save_dir)
        res.save_to_json(save_path=os.path.join(save_dir, "res.json"))
    
    print(f"完成。结果保存在 {save_dir}")

if __name__ == "__main__":
    run_test()
