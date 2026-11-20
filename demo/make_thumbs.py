"""Small assets for render_search.py and the attribution list (runs where the 256 bank images are).

python demo/make_thumbs.py --pick demo/data/coco_pick.json --image_dir /content/demo_imgs --search demo/data/search_glance.json
-> demo/data/search_sprite.jpg   16 x 16 tiles, 64 px centre crops, bank order
   demo/data/search_large/<id>.jpg  360 px wide copies of images with the target or selected by Glance
   demo/data/credits.csv         COCO id, Flickr URL, licence of every image the GIFs show
"""
import argparse, csv, json, os
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--search", required=True)
    ap.add_argument("--tile", type=int, default=64)
    a = ap.parse_args()
    out = os.path.dirname(a.pick)
    p = json.load(open(a.pick))
    bank = p["search"]["bank"]
    s = json.load(open(a.search))
    T = a.tile
    sheet = Image.new("RGB", (16 * T, 16 * T))
    for k, r in enumerate(bank):
        im = Image.open(os.path.join(a.image_dir, r["file"])).convert("RGB")
        w, h = im.size
        m = min(w, h)
        im = im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m)).resize((T, T), Image.LANCZOS)
        sheet.paste(im, ((k % 16) * T, (k // 16) * T))
    sheet.save(os.path.join(out, "search_sprite.jpg"), quality=88)
    keep = set()
    for q in s["queries"].values():
        keep |= set(q["selected"]) | set(q["truth_present"])
    os.makedirs(os.path.join(out, "search_large"), exist_ok=True)
    for k in sorted(keep):
        r = bank[k]
        im = Image.open(os.path.join(a.image_dir, r["file"])).convert("RGB")
        im = im.resize((360, round(im.height * 360 / im.width)), Image.LANCZOS)
        im.save(os.path.join(out, "search_large", f"{r['id']}.jpg"), quality=88)
    race = json.load(open(os.path.join(os.path.dirname(out), "race_questions.json")))
    rrec = next(r for r in p["race"] if r["id"] == race["coco_id"])
    with open(os.path.join(out, "credits.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["gif", "coco_id", "coco_url", "flickr_url", "license", "license_url"])
        for gif, recs in (("glance_vs_vllm.gif", [rrec]), ("search_256.gif", bank)):
            for r in recs:
                wr.writerow([gif, r["id"], r["coco_url"], r["flickr_url"], r["license"], r["license_url"]])
    print("sprite +", len(keep), "large images + credits")


if __name__ == "__main__":
    main()
