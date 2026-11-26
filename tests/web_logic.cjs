// Code-level DOM substitute, not a browser visual or accessibility acceptance test.
const fs=require('fs'),vm=require('vm'),path=require('path'),assert=require('assert');
const os=require('os'),{execFileSync}=require('child_process');
function encodedArchive(log){
 const [header,...frames]=log.trimEnd().split('\n').map(JSON.parse);
 const fixed=Buffer.alloc(412), records=[];
 fixed.write('QQGame JQS');
 const coordinate=n=>[header.nodes[n][0],16-header.nodes[n][1]];
 for(let seat=0;seat<4;seat++){
  const offset=60+88*((seat+3)%4);fixed[offset-28]=seat;
  for(let p=0;p<30;p++){
   const rank=frames[0].truth[seat*30+p];
   fixed[offset+Math.floor(p/5)*5+4-p%5]=rank<0?0:rank<3?rank+2:16-rank;
  }
 }
 for(let i=1;i<frames.length;i++){
  const row=frames[i],[,from,to,collision,out,reveal]=row.event,before=frames[i-1];
  const flags=[];
  for(let seat=0;seat<4;seat++)if(reveal&(1<<seat)){
   flags.push(...coordinate(before.truth.findIndex((rank,n)=>rank===0&&before.seats[n]===seat)));
  }
  records.push(Buffer.from([95,collision,...coordinate(from),...coordinate(to),...flags,0,0,0,0].slice(0,10)));
  for(let seat=0;seat<4;seat++)if(out&(1<<seat)){
   const flag=before.truth[to]===0&&before.seats[to]===seat&&(collision===1||collision===3);
   records.push(Buffer.from([245,3,seat,flag?1:2,255,0,0,0,0,0]));
  }
 }
 assert([1,2,3].includes(frames.at(-1).result));
 records.push(Buffer.from([245,5,0,frames.at(-1).result===3?2:4,255,0,0,0,0,0]));
 fixed.writeUInt16LE(records.length,26);
 return Buffer.concat([fixed,...records]);
}
class Element{
 constructor(){this.children=[];this.value='0';this.disabled=false;this.hidden=false;this.textContent='';this.attrs={};}
 appendChild(e){this.children.push(e);}replaceChildren(){this.children=[];}setAttribute(k,v){this.attrs[k]=v;}
}
const els=new Map(),get=id=>{if(!els.has(id))els.set(id,new Element());return els.get(id);};
let tick;
const context=vm.createContext({console,document:{getElementById:get,createElement:()=>new Element(),createElementNS:()=>new Element()},window:{},setInterval:f=>{tick=f;return 1;},clearInterval:()=>{tick=null;}});
const root=path.resolve(__dirname,'..');
const fixtureDir=fs.mkdtempSync(path.join(os.tmpdir(),'junqi-web-fixture-'));
const fixture=path.join(fixtureDir,'game.jsonl');
try{
 execFileSync(path.join(root,'build/junqi.exe'),['simulate','42',fixture,
  path.join(root,'config/ruleset.json'),path.join(root,'config/observation.json')]);
 context.window.JUNQI_LOG=fs.readFileSync(fixture,'utf8');
}finally{
 if(fs.existsSync(fixture))fs.unlinkSync(fixture);
 fs.rmdirSync(fixtureDir);
}
vm.runInContext(fs.readFileSync(path.join(root,'web/review.js'),'utf8'),context);
vm.runInContext(fs.readFileSync(path.join(root,'web/app.js'),'utf8'),context);
assert(!get('play').disabled);assert.match(get('position').textContent,/第 0/);
assert.match(get('policy').textContent,/随机合法走法/);
assert.equal(get('materialLegend').textContent,'南北队 50 子 · 东西队 50 子');
assert.match(get('materialSeats').children[1].children[1].textContent,/暗子 25/);
assert.doesNotMatch(get('materialSeats').children[1].children[1].textContent,/合法着|能走/);
get('nextExchange').onclick();assert.match(get('exchangeText').textContent,/存活|同消/);
get('decisionView').onclick();assert(!get('wholeGame').checked&&get('nextExchange').disabled);
get('returnAction').onclick();assert.match(get('exchangeText').textContent,/存活|同消/);
get('first').onclick();get('view').value='0';get('view').onchange();
const hidden=()=>get('board').children.filter(e=>e.textContent==='暗').length;
assert.equal(hidden(),75);get('view').value='4';get('view').onchange();assert.equal(hidden(),0);
for(let v=0;v<4;v++){get('view').value=String(v);get('view').onchange();assert.equal(hidden(),75);}
get('next').onclick();assert.match(get('position').textContent,/第 1 /);
get('prev').onclick();assert.match(get('position').textContent,/第 0 /);
get('play').onclick();assert.equal(get('play').textContent,'暂停');tick();assert.match(get('position').textContent,/第 1 /);
get('play').onclick();assert.equal(tick,null);get('last').onclick();assert.equal(get('result').textContent,'和棋');
get('first').onclick();get('timeline').oninput({target:{value:'5'}});assert.match(get('position').textContent,/第 5 /);
get('timeline').oninput({target:{value:'503'}});
assert.match(get('teams').textContent,/北方 无合法着出局/);
assert.match(get('event').textContent,/北方无合法着出局/);
get('prev').onclick();
assert.match(get('teams').textContent,/北方 在场/);
assert(!get('event').textContent.includes('北方无合法着'));
assert.deepEqual(JSON.parse(vm.runInContext(`JSON.stringify(readExits([
 {truth:[-1,0],seats:[-1,2]}, {event:[0,0,1,1,6,0]}
 ],true))`,context)),[null,{ply:1,reason:'无合法着'},{ply:1,reason:'军旗被夺'}]);
assert.equal(vm.runInContext(`readExits([
 {truth:[-1,0],seats:[-1,2]}, {event:[0,0,1,1,4,0]}
 ],false)[2].reason`,context),'无合法着');
get('timeline').oninput({target:{value:'5'}});
get('next').onclick();get('decisionView').onclick();assert.match(get('position').textContent,/第 5 /);
const mock=JSON.parse(JSON.stringify(vm.runInContext('frames',context)));
mock[1].decision={policy:'heuristic-1',viewer:mock[1].event[0],based_on_seq:0,objective:0,position:3,exposure:0,total:3};
mock[1].event[5]=3;
context.mock=mock;
get('view').value=String((mock[1].event[0]+1)%4);
vm.runInContext('frames=mock;index=1;render()',context);
assert.doesNotMatch(get('decision').textContent,/合计/);
assert.match(get('event').textContent,/南方、东方司令阵亡，触发亮旗/);
get('view').value=String(mock[1].event[0]);get('view').onchange();
assert.match(get('decision').textContent,/合计 3/);
get('view').value='4';get('view').onchange();assert.match(get('decision').textContent,/合计 3/);
get('decisionView').onclick();assert.equal(get('view').value,String(mock[1].event[0]));
assert.match(get('position').textContent,/第 0 /);
assert.doesNotMatch(get('event').textContent,/触发亮旗/);
const singleFile=path.join(fs.mkdtempSync(path.join(os.tmpdir(),'junqi-web-')),'game.jsonl');
const capFile=path.join(path.dirname(singleFile),'cap.json');
try{
 for(let seed=43;seed<55;seed++){
  execFileSync(path.join(root,'build/junqi.exe'),['simulate-single',String(seed),singleFile,
   path.join(root,'config/ruleset.json'),path.join(root,'config/observation.json')]);
  const text=fs.readFileSync(singleFile,'utf8');fs.unlinkSync(singleFile);
  if(JSON.parse(text.trimEnd().split('\n').at(-1)).result===2){context.singleDoc=text;break;}
 }
 assert(context.singleDoc,'fixture needs a north win');
 get('view').value='1';vm.runInContext('load(singleDoc)',context);
 assert.equal(get('view').value,'0');assert(get('view-east').disabled&&get('view-west').hidden);
 assert.equal(hidden(),25);assert.match(get('mode').textContent,/单挑暗棋/);
 assert.equal(get('materialLegend').textContent,'南方 25 子 · 北方 25 子');
 assert.match(get('teams').textContent,/东方 无参赛者/);assert.match(get('teams').textContent,/西方 无参赛者/);
 assert.equal(get('board').children.filter(e=>e.attrs['stroke-width']===.10).length,
  vm.runInContext('header.rails.length',context));
 get('view').value='4';get('view').onchange();assert.equal(hidden(),0);
 get('next').onclick();get('decisionView').onclick();assert.equal(hidden(),25);
 get('last').onclick();assert.equal(get('result').textContent,'北方获胜');
 get('view').value='2';get('first').onclick();assert.equal(hidden(),25);
 const lines=context.singleDoc.trimEnd().split('\n'); const bad=JSON.parse(lines[1]);bad.turn=1;
 lines[1]=JSON.stringify(bad);context.badSingle=lines.join('\n')+'\n';
 assert.throws(()=>vm.runInContext('load(badSingle)',context),/未参赛座位/);
 vm.runInContext('load(window.JUNQI_LOG)',context);
 assert(!get('view-east').disabled&&!get('view-west').hidden);assert.equal(hidden(),75);
 for(const [command, title, hiddenRanks] of [['simulate-dual','双明',50],['simulate-full','全明对局',0]]){
  execFileSync(path.join(root,'build/junqi.exe'),[command,'42',singleFile,
   path.join(root,'config/ruleset.json'),path.join(root,'config/observation.json')]);
  context.nativeDoc=fs.readFileSync(singleFile,'utf8');fs.unlinkSync(singleFile);
  vm.runInContext('load(nativeDoc)',context);
  for(let v=0;v<4;v++){
   get('view').value=String(v);get('view').onchange();
   assert.equal(hidden(),hiddenRanks);assert(get('mode').textContent.endsWith(title));
  }
  get('view').value='4';get('next').onclick();assert.equal(hidden(),0);
  get('decisionView').onclick();assert.equal(hidden(),hiddenRanks);
 }
 const selfplay=path.join(root,'config/ruleset-selfplay.json'), obs=path.join(root,'config/observation.json');
 execFileSync(path.join(root,'build/junqi.exe'),['simulate-single','42',singleFile,selfplay,obs,'flag-route']);
 context.practiceDoc=fs.readFileSync(singleFile,'utf8');fs.unlinkSync(singleFile);
 vm.runInContext('load(practiceDoc)',context);get('next').onclick();
 assert.equal(get('policy').textContent,'夺旗练习棋手');
 assert.match(get('validation').textContent,/无着顺延，可动子耗尽出局 · 连续 70 手无碰撞判和/);
 get('view').value='4';get('view').onchange();assert.match(get('decision').textContent,/flag-route-1/);
 execFileSync(path.join(root,'build/junqi.exe'),['simulate','998123',singleFile,selfplay,obs]);
 context.selfplayDoc=fs.readFileSync(singleFile,'utf8');fs.unlinkSync(singleFile);
 vm.runInContext('load(selfplayDoc)',context);
 const nativeReview=path.join(path.dirname(singleFile),'review.jsonl');
 fs.writeFileSync(singleFile,context.selfplayDoc);
 try{
  execFileSync(path.join(root,'build/junqi.exe'),['review',singleFile,nativeReview]);
  execFileSync(path.join(root,'build/junqi.exe'),['replay',nativeReview]);
  context.reviewDoc=fs.readFileSync(nativeReview,'utf8');
  get('view').value='0';vm.runInContext('load(reviewDoc)',context);
  assert.match(get('materialSeats').children[0].children[1].textContent,/能走 \d+／堵住 \d+/);
  assert.doesNotMatch(get('materialSeats').children[1].children[1].textContent,/能走/);
  assert(!get('flagWatch').textContent.includes('旧日志'));
  const badReview=context.reviewDoc.trimEnd().split('\n');
  const beginning=JSON.parse(badReview[1]);beginning.move_origins[0]++;
  badReview[1]=JSON.stringify(beginning);fs.writeFileSync(singleFile,badReview.join('\n')+'\n');
  assert.throws(()=>execFileSync(path.join(root,'build/junqi.exe'),['replay',singleFile],{stdio:'pipe'}));
 }finally{
  if(fs.existsSync(nativeReview))fs.unlinkSync(nativeReview);
  fs.unlinkSync(singleFile);
 }
 const rows=JSON.parse(JSON.stringify(vm.runInContext('frames',context)));
 const exhausted=rows.findIndex(row=>row.eliminations?.some(exit=>exit.reason==='movable_pieces_exhausted'));
 assert(exhausted>0);
 get('timeline').oninput({target:{value:String(exhausted)}});
 assert.match(get('teams').textContent,/可动子耗尽出局/);
 assert.match(get('event').textContent,/可动子耗尽出局/);
 const skipped=rows.findIndex(row=>row.auto_skipped_mask);
 assert(skipped>0);
 get('timeline').oninput({target:{value:String(skipped)}});
 assert.match(get('event').textContent,/暂时堵塞，自动顺延/);
 const tampered=context.selfplayDoc.trimEnd().split('\n').map(JSON.parse);
 tampered[exhausted+1].eliminations[0].ply++;
 context.invalidExit=tampered.map(row=>JSON.stringify(row)).join('\n')+'\n';
 assert.throws(()=>vm.runInContext('load(invalidExit)',context),/退场原因或手数/);
 const rules=JSON.parse(fs.readFileSync(selfplay,'utf8'));
 rules.experimental['sim.truncation_cap']=1;
 fs.writeFileSync(capFile,JSON.stringify(rules));
 execFileSync(path.join(root,'build/junqi.exe'),['simulate','998123',singleFile,capFile,obs]);
 context.cutoffDoc=fs.readFileSync(singleFile,'utf8');
 vm.runInContext('load(cutoffDoc)',context);get('last').onclick();
 assert.equal(get('result').textContent,'达到实验手数上限，未结算');
 fs.unlinkSync(singleFile);

 const archiveRules=JSON.parse(fs.readFileSync(path.join(root,'config/ruleset.json')));
 Object.assign(archiveRules.experimental,{'turn.order':'cw','elim.no_legal_move':'eliminated_after_involved_move',
  'draw.no_progress.limit':71,'marshal_death.reveal_flag':true});
 fs.writeFileSync(capFile,JSON.stringify(archiveRules));
 const folder=path.dirname(singleFile), observationFile=path.join(folder,'obs.json'),
  jgs=path.join(folder,'fixture.jgs'), imported=path.join(folder,'imported.jsonl');
 try{
  const visibility=JSON.parse(fs.readFileSync(obs));visibility.marshal_reveals_flag=true;
  fs.writeFileSync(observationFile,JSON.stringify(visibility));
  const cli=path.join(root,'build/junqi.exe');
  execFileSync(cli,['simulate','178',singleFile,capFile,observationFile]);
  const fixture=encodedArchive(fs.readFileSync(singleFile,'utf8'));
  fixture.write('privateMark',14);
  fs.writeFileSync(jgs,fixture);
  const args=['import-jgs',jgs,imported,path.join(root,'config/ruleset.json'),obs];
  execFileSync(cli,args);execFileSync(cli,['replay',imported]);
  context.archiveDoc=fs.readFileSync(imported,'utf8');
  assert(!context.archiveDoc.includes(Buffer.from('privateMark').toString('hex')));
  vm.runInContext('load(archiveDoc)',context);get('last').onclick();
  assert.match(get('position').textContent,/原谱 \d+–\d+/);
  assert.match(get('event').textContent,/原谱 \d+ 结束/);
  const importedRows=context.archiveDoc.trimEnd().split('\n').map(JSON.parse);
  const lastFrame=importedRows.at(-1);
  if(lastFrame.result===3)assert.match(get('result').textContent,/原谱和棋 · 静步计数/);
  lastFrame.result=lastFrame.result===3?4:3;
  context.wrongArchiveEnding=importedRows.map(row=>JSON.stringify(row)).join('\n')+'\n';
  assert.throws(()=>vm.runInContext('load(wrongArchiveEnding)',context),/和棋结果与终局记录不一致/);
  assert.match(vm.runInContext("archiveDrawLabel({no_progress:71},{'draw.no_progress.limit':71})",context),
               /静步计数 71 · 达到档案静步阈值/);
  assert.match(vm.runInContext("archiveDrawLabel({no_progress:5},{'draw.no_progress.limit':71})",context),
               /静步计数 5 · 原谱未记载和棋触发原因/);
  fs.unlinkSync(imported);
  const firstSeat=JSON.parse(fs.readFileSync(singleFile,'utf8').split('\n')[0]).first;
  const skipped=(firstSeat+1)%4;
  const beforeFirst=Buffer.concat([fixture.subarray(0,412),
    Buffer.from([245,2,skipped,1,firstSeat,0,0,0,0,0]),fixture.subarray(412)]);
  beforeFirst.writeUInt16LE(fixture.readUInt16LE(26)+1,26);
  fs.writeFileSync(jgs,beforeFirst);
  execFileSync(cli,args);execFileSync(cli,['replay',imported]);
  context.archiveDoc=fs.readFileSync(imported,'utf8');
  const withSkip=JSON.parse(context.archiveDoc.split('\n')[0]);
  assert.deepEqual(withSkip.source_controls[0],[0,1,2,skipped,1]);
  assert.equal(withSkip.source_steps[1],2);
  vm.runInContext('load(archiveDoc)',context);get('first').onclick();
  assert.match(get('event').textContent,/原谱 1.*跳过记录/);
  fs.unlinkSync(imported);
  beforeFirst[416]=(firstSeat+2)%4;
  fs.writeFileSync(jgs,beforeFirst);
  assert.throws(()=>execFileSync(cli,args,{stdio:'pipe'}),/JGS record 2/);
  assert(!fs.existsSync(imported));
  fixture[fixture.length-9]=99;
  fs.writeFileSync(jgs,fixture);
  assert.throws(()=>execFileSync(cli,args,{stdio:'pipe'}),/JGS record.*unsupported JGS control/s);
  assert(!fs.existsSync(imported));
 }finally{
  for(const file of [observationFile,jgs,imported])if(fs.existsSync(file))fs.unlinkSync(file);
 }
}finally{
 if(fs.existsSync(singleFile))fs.unlinkSync(singleFile);
 if(fs.existsSync(capFile))fs.unlinkSync(capFile);
 fs.rmdirSync(path.dirname(singleFile));
}
assert.throws(()=>vm.runInContext('load("broken")',context));
vm.runInContext('fail(new Error("test invalid log"))',context);assert(get('play').disabled);assert.equal(get('board').children.length,0);assert(!get('error').hidden);
console.log('PASS web logic: review navigation, hidden views, strict JGS import, selfplay settlement and rejected input');
