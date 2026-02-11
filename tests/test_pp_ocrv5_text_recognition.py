# PP-OCRv5 文本识别模块测试脚本
import os

# 辅助函数：生成测试图片
def ensure_test_image(path):
    if not os.path.exists(path):
        try:
            from PIL import Image, ImageDraw
            # 识别模型通常输入是裁剪好的文本行小图
            img = Image.new('RGB', (200, 32), color=(255, 255, 255))
            d = ImageDraw.Draw(img)
            d.text((10, 10), "Recognition", fill=(0, 0, 0))
            img.save(path)
            print(f"已生成测试图片: {path}")
        except Exception as e:
            print(f"无法生成图片: {e}")

def run_test():
    image_path = "./general_ocr_rec_001.png"
    ensure_test_image(image_path)

    try:
        from paddleocr import TextRecognition
    except ImportError:
        print("错误: 无法导入 paddleocr.TextRecognition。")
        return

    print("正在初始化 TextRecognition 模型...")
    model = TextRecognition()
    
    print(f"开始识别: {image_path}")
    output = model.predict(input=image_path)
    
    save_dir = "./output/TextRecognition_v5"
    os.makedirs(save_dir, exist_ok=True)

    for res in output:
        res.print()
        res.save_to_img(save_path=save_dir)
        res.save_to_json(save_path=os.path.join(save_dir, "res.json"))

    print(f"完成。结果保存在 {save_dir}")

if __name__ == "__main__":
    run_test()
