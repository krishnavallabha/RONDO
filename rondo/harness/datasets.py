"""External match data: what to download, where it lands, and whether it arrived intact.

Two public datasets stand in for this match while we tune. Neither is the challenge data and
neither is ever committed -- everything goes under data/external/ (gitignored), or wherever
RONDO_EXTERNAL_DIR points.

    IDSSE / Sportec  7 Bundesliga matches, 25 fps, per-frame BallPossession + BallStatus in the
                     positional XML. These are the held-out matches the decision rule is applied
                     to: their labels come from the provider, not from us.
                     Bassek et al. (2024), CC BY, figshare 10.6084/m9.figshare.28196177

    Metrica sample   2 matches, 25 fps, tracking CSV + hand-coded event CSV. No per-frame
                     possession channel, so labels are derived from the events -- which is why
                     this is the convention anchor: Metrica authored both the sample events and
                     this challenge's ground truth, so their idea of when a possession switches
                     and where a dead ball starts is the one to copy.
                     metrica-sports/sample-data, CC BY-NC 4.0

Sizes and md5s are pinned here rather than looked up, so a fetch either reproduces the bytes
every other team member has or fails loudly. The Metrica URL is pinned to a commit for the same
reason -- that repository has no releases.

    python -m rondo.harness fetch                   # everything (~2.6 GB, mostly IDSSE positions)
    python -m rondo.harness fetch --dataset metrica
    python -m rondo.harness fetch --match J03WMX
"""

import hashlib
import os
import shutil
import urllib.request

EXTERNAL_DIR = os.environ.get("RONDO_EXTERNAL_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "external")

FIGSHARE_FILE = "https://ndownloader.figshare.com/files/{}"
METRICA_RAW = ("https://raw.githubusercontent.com/metrica-sports/sample-data/"
               "e706dd506b360d69d9d123d5b8026e7294b13996/data/{}")

# IDSSE, one entry per match: asset -> (figshare file id, md5, bytes). The DFL match id is the
# key; the competition id appears only in the file names, so it is needed nowhere else.
IDSSE = {
    "J03WMX": {
        "matchinfo": (51643475, "bc5d5b21dbcf36aa30f4cd0428309f6b", 11930),
        "events": (51643493, "2d88e180a1f63c7e4157feb980fc7255", 776578),
        "positions": (51643514, "e4273a2adf5dbd9d4ec42def2f01905c", 418524833),
    },
    "J03WN1": {
        "matchinfo": (51643472, "37b0f85588c0acfb995fe633c7ac2e0d", 11722),
        "events": (51643496, "83e8ed7ea8e8ebbbdc61f61ac3534008", 612114),
        "positions": (51643517, "bacb2f609c1d1b0bfc906f6b9d4109f2", 391722803),
    },
    "J03WOH": {
        "matchinfo": (51643478, "b9710781d400dd812f62b0a6e3d7bce8", 11745),
        "events": (51643499, "ab66e51512f39d06f49f27dd3b4c2f40", 635747),
        "positions": (51643520, "3efff25a86a61768e7d253b4bba137a3", 348945715),
    },
    "J03WOY": {
        "matchinfo": (51643481, "7b021e577fca917a63111d55cc29ae5b", 11952),
        "events": (51643502, "64b1be01d6ff6848168efebd56f344b9", 683018),
        "positions": (51643523, "4e1063e0a0ae499920cf68ac1aa3ab4e", 362567859),
    },
    "J03WPY": {
        "matchinfo": (51643487, "65a4568e4f374af94d3c1015cbc7c4f5", 11922),
        "events": (51643505, "f68482441d38723d69f244f896482803", 698898),
        "positions": (51643526, "e56ff1336a334f8da51110fbe7cab764", 371688499),
    },
    "J03WQQ": {
        "matchinfo": (51643484, "5979209af316a914ce57347d0633df6c", 11932),
        "events": (51643508, "b7e9d82af4034539d802319be6c57776", 732186),
        "positions": (51643529, "9105bad987191d3c3c362c1fce9a4fcd", 356219423),
    },
    "J03WR9": {
        "matchinfo": (51643490, "5e2d580d3d8ed97e32959de90d04992b", 11823),
        "events": (51643511, "c13bc5efa5eca4a022cc678210b87d5f", 681050),
        "positions": (51643532, "e060bb7d7a5a3bfbd15bbf9bc631024d", 372934140),
    },
}

# Metrica, one entry per match: asset -> (path inside their repo, md5, bytes). Sample_Game_3 is
# deliberately absent: it is an EPTS export with JSON events, so it would need a second parser
# for no extra labelled minutes. Signals can pull it by hand to test the EPTS loader.
METRICA = {
    "Sample_Game_1": {
        "events": ("Sample_Game_1/Sample_Game_1_RawEventsData.csv",
                   "b46663d153266ad9e8d97183dc545b19", 139446),
        "tracking_home": ("Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
                          "2c9cbe4c4acb75180c71df7f8e77b0a8", 32782961),
        "tracking_away": ("Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
                          "2b2f77617a0d61cab3712d78940feffb", 32782073),
    },
    "Sample_Game_2": {
        "events": ("Sample_Game_2/Sample_Game_2_RawEventsData.csv",
                   "2fb475d62a0a6d5dbde863a98e3935fc", 153897),
        "tracking_home": ("Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
                          "fc7c71a9d90e06c3bfa1b21b1e81fb77", 31890338),
        "tracking_away": ("Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
                          "c398b8dd96e0d0b7f7d1c9e4146f78d1", 29627711),
    },
}

DATASETS = ("idsse", "metrica")

# Both datasets run at 25 fps, the same rate as the tactical video, so a frame of external data
# is directly comparable to a frame of ours and no resampling is needed anywhere downstream.
FPS = 25


def match_ids(dataset):
    """Match keys of one dataset, in a fixed order so reports are stable."""
    return sorted(IDSSE if dataset == "idsse" else METRICA)


def every_match():
    """[(dataset, match_id)] for both datasets, in report order."""
    return [(d, m) for d in DATASETS for m in match_ids(d)]


def path(dataset, match_id, asset, external_dir=None):
    """Where one file lives once fetched. Does not check that it is there."""
    suffix = ".xml" if dataset == "idsse" else ".csv"
    return os.path.join(external_dir or EXTERNAL_DIR, dataset, match_id, asset + suffix)


def assets(dataset, match_id):
    """asset -> (url, md5 or None, expected bytes)."""
    if dataset == "idsse":
        return {name: (FIGSHARE_FILE.format(fid), md5, size)
                for name, (fid, md5, size) in IDSSE[match_id].items()}
    return {name: (METRICA_RAW.format(rel), md5, size)
            for name, (rel, md5, size) in METRICA[match_id].items()}


def have(dataset, match_id, names=None, external_dir=None):
    """True when every wanted asset is on disk at its pinned size."""
    table = assets(dataset, match_id)
    for name in names or table:
        local = path(dataset, match_id, name, external_dir)
        if not os.path.isfile(local) or os.path.getsize(local) != table[name][2]:
            return False
    return True


def require(dataset, match_id, names, external_dir=None):
    """Local paths for the named assets, or an IOError that says how to get them."""
    out = []
    for name in names:
        local = path(dataset, match_id, name, external_dir)
        if not os.path.isfile(local):
            raise IOError(
                "{} {} {} not downloaded:\n  {}\n\nRun: python -m rondo.harness fetch --match {}"
                .format(dataset, match_id, name, local, match_id))
        out.append(local)
    return out


def md5_of(local, chunk=1 << 20):
    """md5 of a file, read in blocks -- the positional XMLs are 400 MB."""
    digest = hashlib.md5()
    with open(local, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_file(url, local, md5=None, size=None, force=False, log=print):
    """Download one file unless it is already there at the right size. Returns the path.

    Written to a .part file and renamed only once it is complete and checked, so an interrupted
    download never leaves something that looks finished. A .part left by an earlier run is
    resumed with a Range request rather than restarted: the positional XMLs are 400 MB and the
    link to figshare is not fast. A server that ignores the Range header answers 200 with the
    whole file, which is handled by starting the .part again.
    """
    if not force and os.path.isfile(local) and (size is None or os.path.getsize(local) == size):
        return local
    os.makedirs(os.path.dirname(local), exist_ok=True)
    part = local + ".part"
    done = os.path.getsize(part) if os.path.isfile(part) and not force else 0
    if size is not None and done >= size:
        done = 0  # a .part at or past the full size is junk from an earlier failure

    request = urllib.request.Request(url)
    if done:
        request.add_header("Range", "bytes={}-".format(done))
        log("  get {} (resuming at {:.0f} MB)".format(os.path.basename(local), done / 1e6))
    else:
        log("  get {}".format(os.path.basename(local)))
    with urllib.request.urlopen(request, timeout=180) as response:
        resumed = response.status == 206
        with open(part, "ab" if resumed else "wb") as handle:
            shutil.copyfileobj(response, handle, 1 << 20)

    got = os.path.getsize(part)
    if size is not None and got != size:
        if got > size:
            os.remove(part)  # longer than pinned: the wrong file, not a short read
            raise IOError("{}: got {} bytes, expected {}".format(url, got, size))
        raise IOError("{}: stopped at {} of {} bytes; run fetch again to resume".format(
            url, got, size))
    if md5 is not None:
        digest = md5_of(part)
        if digest != md5:
            os.remove(part)
            raise IOError("{}: md5 {} != pinned {}".format(url, digest, md5))
    os.replace(part, local)
    return local


def fetch(dataset, match_id, names=None, force=False, external_dir=None, log=print):
    """Download one match. Returns asset -> local path."""
    table = assets(dataset, match_id)
    log("{} {}".format(dataset, match_id))
    out = {}
    for name in names or sorted(table):
        url, md5, size = table[name]
        out[name] = fetch_file(url, path(dataset, match_id, name, external_dir),
                               md5, size, force, log)
    return out


def fetch_all(datasets=DATASETS, matches=None, names=None, force=False, external_dir=None,
              log=print):
    """Download every match of the named datasets, or just the named matches."""
    total = 0
    for dataset in datasets:
        for match_id in match_ids(dataset):
            if matches and match_id not in matches:
                continue
            fetch(dataset, match_id, names, force, external_dir, log)
            total += 1
    log("{} match(es) under {}".format(total, external_dir or EXTERNAL_DIR))
    return total
