# PP-StructureV3 测试脚本
import os

# 辅助函数：生成测试图片
def ensure_test_image(path):
    if not os.path.exists(path):
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (600, 800), color=(255, 255, 255))
            d = ImageDraw.Draw(img)
            # 模拟一个表格
            d.rectangle([50, 50, 550, 300], outline="black")
            d.line([50, 100, 550, 100], fill="black")
            d.line([200, 50, 200, 300], fill="black")
            d.text((60, 60), "Header 1", fill=(0,0,0))
            d.text((210, 60), "Header 2", fill=(0,0,0))
            img.save(path)
            print(f"已生成测试图片: {path}")
        except Exception as e:
            print(f"无法生成图片: {e}")

def run_test():
    # 强制关闭 OneDNN/MKLDNN 以规避 Windows/CPU 上的属性转换未实现报错 todo 没用
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["PADDLE_ENABLE_ONEDNN"] = "0"
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    # 获取当前脚本所在的绝对路径，确保图片生成在 tests 目录下
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "test_images")
    os.makedirs(img_dir, exist_ok=True)
    image_path = os.path.join(img_dir, "pp_structure_v3_demo.png")

    ensure_test_image(image_path)

    try:
        from paddleocr import PPStructureV3
    except ImportError:
        print("错误: 无法导入 paddleocr.PPStructureV3。")
        return

    print("正在初始化 PPStructureV3 模型...")
    pipeline = PPStructureV3( 
        use_doc_orientation_classify=False, 
        use_doc_unwarping=False
    ) 
    
    print(f"开始版面分析: {image_path}")
    output = pipeline.predict(input=image_path) 
    
    save_dir = "output/PP_Structure_v3"
    os.makedirs(save_dir, exist_ok=True)

    for res in output: 
        res.print() 
        res.save_to_json(save_path=save_dir) 
        res.save_to_markdown(save_path=save_dir)

    print(f"完成。结果保存在 {save_dir}")

if __name__ == "__main__":
    run_test()
