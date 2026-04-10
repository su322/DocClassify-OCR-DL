# 测试 PP-OCRv5 模型
# https://www.paddleocr.ai/latest/quick_start.html#__tabbed_2_1
from paddleocr import PaddleOCR 
import os

# 辅助函数：生成测试图片
def ensure_test_image(path):
    if not os.path.exists(path):
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (400, 100), color=(255, 255, 255))
            d = ImageDraw.Draw(img)
            d.text((10, 40), "Hello PaddleOCR Test", fill=(0, 0, 0))
            img.save(path)
            print(f"已自动生成测试图片: {path}")
        except Exception as e:
            print(f"无法生成测试图片: {e}")

def run_test():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "test_images")
    os.makedirs(img_dir, exist_ok=True)
    image_path = os.path.join(img_dir, "general_ocr_002.png")

    ensure_test_image(image_path)
    
    if not os.path.exists(image_path):
        print(f"Error: 测试图片不存在 {image_path}")
        return

    print("正在初始化 PaddleOCR...")
    # 文本检测+文本识别
    ocr = PaddleOCR(
        use_doc_orientation_classify=False, 
        use_doc_unwarping=False, 
        use_textline_orientation=False,
    )
    
    # ocr = PaddleOCR(use_doc_orientation_classify=True, use_doc_unwarping=True) # 文本图像预处理+文本检测+方向分类+文本识别
    # ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False) # 文本检测+文本行方向分类+文本识别
    # ocr = PaddleOCR(
    #     text_detection_model_name="PP-OCRv5_mobile_det",
    #     text_recognition_model_name="PP-OCRv5_mobile_rec",
    #     use_doc_orientation_classify=False,
    #     use_doc_unwarping=False,
    #     use_textline_orientation=False) # 更换 PP-OCRv5_mobile 模型

    print(f"开始识别图片: {image_path}")
    
    try:
        # 尝试使用您提供的 predict 接口
        result = ocr.predict(image_path)
        
        # 确保输出目录存在
        save_dir = os.path.join(current_dir, "output", "PaddleOCR_v5_mobile")
        os.makedirs(save_dir, exist_ok=True)
        
        for res in result: 
            res.print() 
            res.save_to_img(save_dir) 
            res.save_to_json(save_dir)
            
        print(f"识别完成，结果已保存至 {save_dir} 目录")
        
    except AttributeError:
        print("注意: 当前版本的 PaddleOCR 可能没有 predict 方法 (通常用于版面分析/表格还原)。")
        print("正在尝试回退到标准的 ocr() 方法...")
        
        result = ocr.ocr(image_path, cls=False)
        for idx, line in enumerate(result):
            print(f"--- Result {idx} ---")
            for box, (text, score) in line:
                print(f"Text: {text}, Score: {score}")
    except Exception as e:
        print(f"运行出错: {e}")

if __name__ == "__main__":
    run_test()
