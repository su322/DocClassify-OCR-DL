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
    image_path = "./pp_structure_v3_demo.png"
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

# /Users/suyuhang/Documents/GitHub/DocClassify-OCR-DL/.venv/bin/python /Users/suyuhang/Documents/GitHub/DocClassify-OCR-DL/tests/test_pp_structurev3.py
# 已生成测试图片: ./pp_structure_v3_demo.png
# Checking connectivity to the model hosters, this may take a while. To bypass this check, set `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` to `True`.
# 正在初始化 PPStructureV3 模型...
# /Users/suyuhang/Documents/GitHub/DocClassify-OCR-DL/.venv/lib/python3.12/site-packages/paddle/utils/cpp_extension/extension_utils.py:712: UserWarning: No ccache found. Please be aware that recompiling all source files may be required. You can download and install ccache from: https://github.com/ccache/ccache/blob/master/doc/INSTALL.md
#   warnings.warn(warning_message)
# Traceback (most recent call last):
#   File "/Users/suyuhang/Documents/GitHub/DocClassify-OCR-DL/.venv/lib/python3.12/site-packages/paddleocr/_pipelines/base.py", line 105, in _create_paddlex_pipeline
#     return create_pipeline(config=self._merged_paddlex_config, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/Users/suyuhang/Documents/GitHub/DocClassify-OCR-DL/.venv/lib/python3.12/site-packages/paddlex/inference/pipelines/__init__.py", line 168, in create_pipeline
#     pipeline = BasePipeline.get(pipeline_name)(
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/Users/suyuhang/Documents/GitHub/DocClassify-OCR-DL/.venv/lib/python3.12/site-packages/paddlex/utils/deps.py", line 207, in _wrapper
#     require_extra(extra, obj_name=pipeline_name, alt=alt)
#   File "/Users/suyuhang/Documents/GitHub/DocClassify-OCR-DL/.venv/lib/python3.12/site-packages/paddlex/utils/deps.py", line 200, in require_extra
#     raise DependencyError(msg)
# paddlex.utils.deps.DependencyError: `PP-StructureV3` requires additional dependencies. To install them, run `pip install "paddlex[ocr]==<PADDLEX_VERSION>"` if you’re installing `paddlex` from an index, or `pip install -e "/path/to/PaddleX[ocr]"` if you’re installing `paddlex` locally.
#
# The above exception was the direct cause of the following exception:
#
# Traceback (most recent call last):
#   File "/Users/suyuhang/Documents/GitHub/DocClassify-OCR-DL/tests/test_pp_structurev3.py", line 52, in <module>
#     run_test()
#   File "/Users/suyuhang/Documents/GitHub/DocClassify-OCR-DL/tests/test_pp_structurev3.py", line 33, in run_test
#     pipeline = PPStructureV3(
#                ^^^^^^^^^^^^^^
#   File "/Users/suyuhang/Documents/GitHub/DocClassify-OCR-DL/.venv/lib/python3.12/site-packages/paddleocr/_pipelines/pp_structurev3.py", line 139, in __init__
#     super().__init__(**kwargs)
#   File "/Users/suyuhang/Documents/GitHub/DocClassify-OCR-DL/.venv/lib/python3.12/site-packages/paddleocr/_pipelines/base.py", line 67, in __init__
#     self.paddlex_pipeline = self._create_paddlex_pipeline()
#                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/Users/suyuhang/Documents/GitHub/DocClassify-OCR-DL/.venv/lib/python3.12/site-packages/paddleocr/_pipelines/base.py", line 107, in _create_paddlex_pipeline
#     raise RuntimeError(
# RuntimeError: A dependency error occurred during pipeline creation. Please refer to the installation documentation to ensure all required dependencies are installed.
#
# 进程已结束，退出代码为 1
