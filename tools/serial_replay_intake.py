"""Index local replay sources without exporting names, comments, or hidden layouts."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import os
import struct
import sys
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from audit_human_timing import sanitized_tape
from audit_qq_archive import INPUT, LIMIT, ROOT, decode, initial_board, public_records
from serial_worker import write_receipt
from local_archive_inputs import archive_value

QIPAN=Path(os.environ.get("JUNQI_LOCAL_COLLECTION_ROOT", ROOT/"data/local-collection"))
DOWNLOADS=Path(os.environ.get("JUNQI_LOCAL_ARCHIVE_ROOT", ROOT/"data/local-archive"))
OLD=(DOWNLOADS/"replays",DOWNLOADS/"annotated")
REPLAYS={".jqh",".jgs",".jqr",".jqg"}
PLAYABLE={".jqh",".jgs"}
TOPICS=("炸","工兵","军旗","回防","配合","侧翼","试探","残局","调动","布局")
QH_RANK={1:10,2:9,3:8,4:7,5:6,6:5,7:4,8:3,9:0,10:11,11:1,12:2}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def hashes(board, moves):
    def orient(x,y,form):
        if form&4: x=16-x
        for _ in range(form&3): x,y=16-y,x
        return 17*y+x
    if any(not 0<=q<=16 for move in moves for q in move):
        raise ValueError("coordinate")
    encoded=[tuple(sorted((orient(x,y,i),rank) for (x,y),rank in board.items())) for i in range(8)]
    family=min(encoded)
    path=min(tuple((orient(x,y,i),orient(xx,yy,i)) for x,y,xx,yy in moves)
             for i,value in enumerate(encoded) if value==family)
    return digest(repr(family).encode("ascii")),digest(repr((family,path)).encode("ascii"))


def jqh(raw):
    if len(raw) not in (5368,5384) or raw[:8]!=bytes.fromhex("474cea03ce07c602"):
        raise ValueError("envelope")
    data=raw[:101]+bytes(((raw[i]^0x13)-raw[i-1])&255 for i in range(101,5368))
    count=struct.unpack_from("<I",data,200)[0]
    if not 0<count<=1002 or 204+4*count>4212:
        raise ValueError("move_count")
    try:
        board={(i%17,i//17):QH_RANK[val%100] for i,val in enumerate(struct.unpack_from("<289i",data,4212)) if val}
    except KeyError as exc:
        raise ValueError("rank_code") from exc
    if len(board)!=100:
        raise ValueError("starting_population")
    rows=list(struct.iter_unpack("<4B",data[204:204+4*count]))
    return board,rows


def jgs(raw):
    _,layouts,_,events=decode(raw)
    board={p:rank-2 if rank<5 else 16-rank for p,(_,rank) in initial_board(layouts).items()}
    if any(e[0] not in (95,245) for e in events):
        raise ValueError("event_type")
    moves=[(e[2],16-e[3],e[4],16-e[5]) for e in events if e[0]==95]
    return board,moves,events


def source_files(root,counters):
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.is_symlink(): continue
        ext=p.suffix.lower()
        if ext in REPLAYS:
            if p.stat().st_size<=LIMIT: yield ext,p.read_bytes(),False
            else: counters["oversize"]+=1
        elif ext==".zip":
            try:
                with ZipFile(p) as archive:
                    for member in archive.infolist():
                        suffix=Path(member.filename).suffix.lower()
                        if member.is_dir() or suffix not in REPLAYS: continue
                        if not 0<member.file_size<=LIMIT:
                            counters["oversize_zip_entry"]+=1
                            continue
                        with archive.open(member) as stream:
                            yield suffix,stream.read(LIMIT+1),True
            except (BadZipFile,RuntimeError,NotImplementedError, OSError):
                counters["unreadable_zip"]+=1


def attachments():
    docs, files, seen = [], [], set()
    word="{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    series=archive_value("intake_series")
    for p in sorted(DOWNLOADS.glob("*.jqh")):
        if any(term in p.name for term in series):
            raw=p.read_bytes()
            files.append((".jqh",raw,False))
            path=p.with_suffix(".docx")
            if not path.is_file(): continue
            try:
                with ZipFile(path) as document:
                    if document.getinfo("word/document.xml").file_size>2_000_000: raise ValueError("document_limit")
                    xml=ElementTree.fromstring(document.read("word/document.xml"))
                lines=["".join(n.text or "" for n in v.iter(word+"t")).strip() for v in xml.iter(word+"p")]
                texts=[v for v in lines if v]
                text_id=digest("\n".join(texts).encode("utf8"))
                docs.append(dict(replay=digest(raw),annotation=text_id,duplicate_annotation=text_id in seen,
                                 paragraphs=len(texts),topics={v:sum(line.count(v) for line in texts) for v in TOPICS}))
                seen.add(text_id)
            except (BadZipFile,KeyError,ElementTree.ParseError,ValueError):
                docs.append(dict(replay=digest(raw),document="unreadable"))
    return files,docs


def catalog():
    sources={str(i):Counter() for i in range(4)}
    old_bytes,old_paths,old_families=set(),set(),set()
    references=json.loads(INPUT.read_text(encoding="utf-8-sig"))
    archived=[(".jgs",raw,False) for raw in public_records(Path(references["corpus"]))]
    for directory in OLD:
        archived.extend(source_files(directory,sources["0"]))
    supplemental=Path(references["supplemental"])
    known_downloads={supplemental,*DOWNLOADS.glob("*.jgs")}
    archived.extend((".jgs",p.read_bytes(),False) for p in sorted(known_downloads) if p.is_file() and p.stat().st_size<=LIMIT)
    def parse(ext,raw):
        if ext==".jqh":
            board,rows=jqh(raw)
            return board,[r for r in rows if r[0]<=16],rows
        return jgs(raw)
    for ext,raw,_ in archived:
        h=digest(raw)
        if h in old_bytes: continue
        old_bytes.add(h)
        if ext not in PLAYABLE: continue
        try:
            board,moves,_=parse(ext,raw)
            family,path=hashes(board,moves)
            old_paths.add(path);old_families.add(family)
        except ValueError:
            sources["0"][ext+"/undecoded"]+=1

    files,docs=attachments()
    records,wire,seen={},[],{}
    for source,batch in (("1",files),("2",source_files(QIPAN,sources["3"]))):
        for ext,raw,zipped in batch:
            tally=sources[source]
            tally[ext+"/files"]+=1
            tally[ext+"/zip_entries"]+=int(zipped)
            h=digest(raw)
            if h in old_bytes: tally[ext+"/byte_in_reference"]+=1
            if ext not in PLAYABLE:
                tally[ext+"/unique_unverified"]+=int(h not in seen and h not in old_bytes)
            if h in seen:
                tally[ext+"/byte_repeat"]+=1
                continue
            seen[h]=ext
            if ext not in PLAYABLE: continue
            try:
                board,moves,rows=parse(ext,raw)
                family,path=hashes(board,moves)
            except ValueError as error:
                tally[ext+"/rejected/"+str(error)]+=1
                continue
            record=dict(key=h,family=family,trajectory=path,plies=len(moves),format=ext,source=source,
                        prior_layout=family in old_families,prior_trajectory=path in old_paths,prior_bytes=h in old_bytes)
            records[h]=record
            if ext==".jgs":
                wire.append(f"G {h} {sanitized_tape(raw,rows)}")
            else:
                fields=["H",h,len(rows),*(v for (x,y),rank in sorted(board.items()) for v in (x,y,rank)),
                        *(v for row in rows for v in row)]
                wire.append(" ".join(map(str,fields)))
    result=dict(reference=dict(bytes=len(old_bytes),trajectories=len(old_paths),families=len(old_families)),
                sources={k:dict(v) for k,v in sources.items()},notes=docs,records=list(records.values()))
    return result,"\n".join(wire)+"\n"


def main(audit):
    path=ROOT/"build/replay-intake"/os.environ["JUNQI_SERIAL_KEY"]
    snapshot,wire=catalog()
    if audit:
        if snapshot!=json.loads((path/"manifest.json").read_text(encoding="utf8")) or wire!=(path/"games.txt").read_text(encoding="utf8"):
            raise ValueError("replay sources changed")
    else:
        path.mkdir(parents=True,exist_ok=False)
        write_receipt(path/"manifest.json",snapshot)
        (path/"games.txt").write_text(wire,encoding="utf8")
    rows=snapshot["records"]
    print(json.dumps(dict(scanned=snapshot["sources"],prior=snapshot["reference"],decoded=len(rows),
                          trajectories=len({x["trajectory"] for x in rows}),paired_notes=len(snapshot["notes"])),ensure_ascii=False))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    if sys.argv[1:] not in ([],["audit"]): raise ValueError("expected audit or no argument")
    main(bool(sys.argv[1:]))
