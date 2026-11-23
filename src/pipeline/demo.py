"""Demo CLI: chat -> chat tersensor + warning

uv run python src/pipeline/demo.py                       # interaktif
uv run python src/pipeline/demo.py --text "u r 1d1ot"    # sekali jalan
"""
import argparse

from moderator import Moderator

HELP = "perintah: /balanced  /strict  /detail  /quit"


def show(out, mode, detail):
    print(f"  censored : {out['censored']}")
    print(f"  warning  : {out['message'] or '-'}")
    if not detail:
        return
    print(f"  slot raw : {out['src']}")
    print(f"  normal   : {out['norm']}" + ("" if out["t1_valid"] else "  (slot tidak cocok, pakai raw)"))
    verdict = "trigger" if out["trigger"] else "aman"
    print(f"  p_toxic  : {out['p_toxic']:.3f} (ambang {mode} {out['threshold']:.2f}) -> {verdict}")
    if out["tags"]:
        print("  tags     : " + " ".join(f"{w}/{t}" for w, t in out["tags"]))
    ms = out["ms"]
    print(f"  latency  : t1 {ms['t1']} ms, t2 {ms['t2']} ms, t3 {ms['t3']} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", nargs="+", help="satu atau beberapa pesan, tanpa mode interaktif")
    ap.add_argument("--mode", choices=["balanced", "strict"], default="balanced")
    ap.add_argument("--device", default=None, help="cuda atau cpu")
    ap.add_argument("--brief", action="store_true", help="tanpa detail tiap tahap")
    a = ap.parse_args()

    print("memuat model...", flush=True)
    mod = Moderator(a.device, a.mode)
    mode, detail = a.mode, not a.brief
    mod.moderate("warm up")
    print(f"siap di {mod.device}, mode {mode}")

    if a.text:
        for text in a.text:
            print(f"> {text}")
            show(mod.moderate(text), mode, detail)
        return

    print(HELP)
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text in ("/balanced", "/strict"):
            mode = text[1:]
            mod.set_mode(mode)
            print(f"  mode {mode}")
            continue
        if text == "/detail":
            detail = not detail
            print(f"  detail {'on' if detail else 'off'}")
            continue
        if text.startswith("/"):
            print(f"  {HELP}")
            continue
        show(mod.moderate(text), mode, detail)


if __name__ == "__main__":
    main()
