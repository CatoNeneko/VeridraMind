
def main() -> None:
    """从源码项目启动推理，与直接执行项目的eval.py使用同一入口。"""
    from pathlib import Path
    import runpy
    import sys



    root = Path(__file__).resolve().parents[2]
    entry = root / "eval.py"
    if not entry.is_file():
        raise SystemExit("此命令需要MokioMind源码目录，请从项目环境直接运行eval.py。")

    sys.path.insert(0, str(root))

    runpy.run_path(str(entry), run_name="__main__")
