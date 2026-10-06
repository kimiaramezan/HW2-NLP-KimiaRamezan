from collections import defaultdict
import math


def entity_type(entity):
    """Return the entity type from its first mention."""
    return entity[0].get("type", "UNK") if entity else "UNK"


def entity_names(entity):
    """Return non-empty, normalized lowercased surface forms."""
    names = []
    for m in entity:
        name = str(m.get("name", "")).strip().lower()
        if name:
            names.append(name)
    return names


def canonical_name(entity):
    names = entity_names(entity)
    if not names:
        return ""
    # Prefer the longest available mention as a stable surface form.
    return max(names, key=lambda x: (len(x.split()), len(x)))


def mention_positions(entity):
    out = []
    for m in entity:
        try:
            sent_id = int(m["sent_id"])
            start, end = map(int, m["pos"])
            out.append((sent_id, start, end))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def min_sentence_distance(head, tail):
    hp = mention_positions(head)
    tp = mention_positions(tail)
    if not hp or not tp:
        return 99
    return min(abs(h[0] - t[0]) for h in hp for t in tp)


def closest_mention_pair(head, tail):
    """Pick the mention pair with smallest sentence/token distance."""
    hp = mention_positions(head)
    tp = mention_positions(tail)
    if not hp or not tp:
        return None, None

    def key(pair):
        h, t = pair
        sent_dist = abs(h[0] - t[0])
        if sent_dist == 0:
            token_dist = min(abs(h[1] - t[2]), abs(t[1] - h[2]))
        else:
            token_dist = 999
        return (sent_dist, token_dist)

    return min(((h, t) for h in hp for t in tp), key=key)


def distance_bucket(n):
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if n <= 4:
        return "3-4"
    return "5+"


def token_distance_bucket(n):
    if n <= 0:
        return "overlap"
    if n <= 2:
        return "1-2"
    if n <= 5:
        return "3-5"
    if n <= 10:
        return "6-10"
    return "11+"


def safe_token(sent, i, default="<BOUNDARY>"):
    if 0 <= i < len(sent):
        return sent[i].lower()
    return default


def pair_features(doc, h_idx, t_idx):
    """Feature dictionary for one ordered entity pair."""
    entities = doc["vertexSet"]
    head = entities[h_idx]
    tail = entities[t_idx]

    ht = entity_type(head)
    tt = entity_type(tail)
    hn = canonical_name(head)
    tn = canonical_name(tail)

    h_mentions = mention_positions(head)
    t_mentions = mention_positions(tail)
    h_closest, t_closest = closest_mention_pair(head, tail)

    sent_dist = min_sentence_distance(head, tail)

    f = {
        "bias": 1.0,
        "h_type": ht,
        "t_type": tt,
        "type_pair": f"{ht}->{tt}",
        "h_name": hn,
        "t_name": tn,
        "name_pair": f"{hn}=>{tn}",
        "same_name": bool(hn and hn == tn),
        "h_mention_count": min(len(head), 5),
        "t_mention_count": min(len(tail), 5),
        "sent_dist": distance_bucket(sent_dist),
        "same_sentence": sent_dist == 0,
        "h_before_t_entity_index": h_idx < t_idx,
        "h_in_title": bool(hn and hn in doc.get("title", "").lower()),
        "t_in_title": bool(tn and tn in doc.get("title", "").lower()),
    }

    # Entity surface-form components.
    hparts = hn.split()
    if hparts:
        f["h_first"] = hparts[0]
        f["h_last"] = hparts[-1]
        f["h_words"] = min(len(hparts), 6)

    tparts = tn.split()
    if tparts:
        f["t_first"] = tparts[0]
        f["t_last"] = tparts[-1]
        f["t_words"] = min(len(tparts), 6)

    # Features from the closest mention pair.
    if h_closest is not None and t_closest is not None:
        hs, ha, hb = h_closest
        ts, ta, tb = t_closest

        if hs < len(doc["sents"]):
            hsent = doc["sents"][hs]
            f["h_left1"] = safe_token(hsent, ha - 1)
            f["h_left2"] = safe_token(hsent, ha - 2)
            f["h_right1"] = safe_token(hsent, hb)
            f["h_right2"] = safe_token(hsent, hb + 1)

        if ts < len(doc["sents"]):
            tsent = doc["sents"][ts]
            f["t_left1"] = safe_token(tsent, ta - 1)
            f["t_left2"] = safe_token(tsent, ta - 2)
            f["t_right1"] = safe_token(tsent, tb)
            f["t_right2"] = safe_token(tsent, tb + 1)

        if hs == ts and hs < len(doc["sents"]):
            sent = doc["sents"][hs]
            if hb <= ta:
                between = sent[hb:ta]
                f["mention_order"] = "H_BEFORE_T"
                gap = ta - hb
            elif tb <= ha:
                between = sent[tb:ha]
                f["mention_order"] = "T_BEFORE_H"
                gap = ha - tb
            else:
                between = []
                f["mention_order"] = "OVERLAP"
                gap = 0

            f["token_gap"] = token_distance_bucket(gap)
            if between:
                lower_between = [x.lower() for x in between]
                f["between_first"] = lower_between[0]
                f["between_last"] = lower_between[-1]
                f["between_len"] = min(len(lower_between), 12)

                # A small bag of between-mention lexical cues.
                for tok in lower_between[:12]:
                    f[f"between_tok={tok}"] = True

                for a, b in zip(lower_between[:11], lower_between[1:12]):
                    f[f"between_bigram={a}_{b}"] = True

    return f


def gold_pair_relations(doc):
    """Map (head, tail) pairs to the set of gold relation ids."""
    pair_to_rels = defaultdict(set)
    for lab in doc.get("labels", []):
        pair_to_rels[(int(lab["h"]), int(lab["t"]))].add(str(lab["r"]))
    return pair_to_rels


def all_ordered_pairs(doc):
    n = len(doc.get("vertexSet", []))
    for h in range(n):
        for t in range(n):
            if h != t:
                yield h, t
