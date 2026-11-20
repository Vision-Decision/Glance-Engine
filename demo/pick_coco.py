"""Pick redistributable COCO val2017 images for the README demos (runs where the annotations are).

python demo/pick_coco.py --ann /content/coco/annotations/instances_val2017.json --out demo/data/coco_pick.json

Only images whose Flickr licence allows redistribution with attribution and derivatives:
  4 = Attribution License (CC BY 2.0), 7 = No known copyright restrictions, 8 = United States Government Work.
race    candidate busy scenes for the 32-question demo (many objects, many categories)
search  a 256-image bank: TARGET is present in a few images; HELD categories are absent from every bank image and
        serve as absent queries (most calibrate the "none" rule, a few are held out as test queries)
"""
import argparse, collections, json, random

OK_LICENSES = {4, 7, 8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="giraffe")
    ap.add_argument("--n_target", type=int, default=4)
    ap.add_argument("--n_held", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = json.load(open(a.ann))
    lic = {l["id"]: l for l in d["licenses"]}
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {i["id"]: i for i in d["images"] if i["license"] in OK_LICENSES}
    anns = collections.defaultdict(list)
    for x in d["annotations"]:
        if x["image_id"] in imgs and not x.get("iscrowd"):
            anns[x["image_id"]].append(x)
    print("redistributable images", len(imgs), collections.Counter(i["license"] for i in imgs.values()))

    def rec(i):
        im = imgs[i]
        return dict(id=i, file=im["file_name"], coco_url=im["coco_url"], flickr_url=im.get("flickr_url"),
                    license=lic[im["license"]]["name"], license_url=lic[im["license"]]["url"],
                    width=im["width"], height=im["height"],
                    objects=[dict(cat=cats[x["category_id"]], bbox=x["bbox"]) for x in anns[i]])

    # race: busy scenes with many large-ish objects of several categories
    def busy(i):
        big = [x for x in anns[i] if x["area"] > 0.004 * imgs[i]["width"] * imgs[i]["height"]]
        return len({x["category_id"] for x in big}) * 2 + min(len(big), 20)
    race = sorted(imgs, key=busy, reverse=True)[:16]

    # search: categories by image frequency among redistributable images
    freq = collections.Counter(c for i in imgs for c in {x["category_id"] for x in anns[i]})
    tid = next(k for k, v in cats.items() if v == a.target)
    held = [c for c, _ in sorted(freq.items(), key=lambda kv: kv[1]) if c != tid][:a.n_held]
    held += [c for c in cats if c not in freq and c != tid][: max(0, a.n_held - len(held))]
    hs = set(held)
    clean = [i for i in imgs if anns[i] and not ({x["category_id"] for x in anns[i]} & hs)]
    with_t = [i for i in clean if any(x["category_id"] == tid for x in anns[i])]
    without = [i for i in clean if i not in set(with_t)]
    rng = random.Random(a.seed)
    rng.shuffle(with_t); rng.shuffle(without)
    bank = with_t[:a.n_target] + without[:256 - a.n_target]
    rng.shuffle(bank)
    held_names = [cats[c] for c in held]
    rng.shuffle(held_names)              # calibration / held-out test split is random, not by frequency
    print("clean", len(clean), "with target", len(with_t), "bank", len(bank))
    print("held:", held_names)
    json.dump(dict(licenses_allowed=sorted(OK_LICENSES), race=[rec(i) for i in race],
                   search=dict(target=a.target, held=held_names, bank=[rec(i) for i in bank])),
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
