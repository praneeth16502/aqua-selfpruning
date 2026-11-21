"""Stdlib-only fallback runner:  python -m tests.run [substring]
Tests are plain `test_*` functions with bare asserts, so pytest collects them unchanged."""
import importlib, pkgutil, sys, time, traceback
import tests


def main(pattern: str = "") -> int:
    passed, failed = 0, []
    for m in sorted(pkgutil.iter_modules(tests.__path__), key=lambda m: m.name):
        if not m.name.startswith("test_"):
            continue
        mod = importlib.import_module(f"tests.{m.name}")
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            if pattern and pattern not in f"{m.name}.{name}":
                continue
            t0 = time.time()
            try:
                getattr(mod, name)()
                passed += 1
            except Exception:
                failed.append(f"{m.name}.{name}")
                print(f"FAIL {m.name}.{name}\n{traceback.format_exc(limit=4)}")
    print(f"\n{passed} passed, {len(failed)} failed")
    for f in failed:
        print("  -", f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
