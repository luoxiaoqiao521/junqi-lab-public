'use strict';
const $=id=>document.getElementById(id), names=['南方','东方','北方','西方'];
const ranks=['军旗','地雷','炸弹','工兵','排长','连长','营长','团长','旅长','师长','军长','司令'];
const colors=['#df8d76','#e3c579','#87b9db','#ad9bd0'];
const exitNames={flag_removed:'军旗被夺',movable_pieces_exhausted:'可动子耗尽',no_legal_move:'无合法着'};
const endNames={no_progress_draw:'静步规则和棋',simultaneous_exhaustion:'双方同时耗尽，未结算',
 no_legal_actions:'全桌无合法动作，未结算',ply_cap:'达到实验手数上限，未结算'};
let header=null, frames=[], exits=[], index=0, timer=null;
let reviewData=null, actionReview=-1, sourceRows=null;
function stop(){if(timer)clearInterval(timer);timer=null;$('play').textContent='播放';}
function fail(e){stop();header=null;frames=[];$('board').replaceChildren();$('replayInsights').hidden=true;$('error').hidden=false;$('error').textContent='无法打开：'+e.message;$('result').textContent='棋谱未通过校验';controls();}
function controls(){for(const id of ['first','prev','play','next','last','timeline'])$(id).disabled=!frames.length;}
function validSettlement(f,i){
 if(!Array.isArray(f.eliminations)||!Number.isInteger(f.auto_skipped_mask)||f.auto_skipped_mask<0||f.auto_skipped_mask>15)
  throw Error('自训结算字段错误');
 let mask=0;
 for(const exit of f.eliminations){
  if(!Number.isInteger(exit.seat)||exit.seat<0||exit.seat>3||exit.ply!==i||
     !Object.hasOwn(exitNames,exit.reason)||(mask&(1<<exit.seat)))throw Error('退场原因或手数错误');
  mask|=1<<exit.seat;
 }
 if(mask!==f.event[4]||(mask&f.auto_skipped_mask))throw Error('退场和自动顺延状态不一致');
 const expected=['none','team_eliminated','team_eliminated','no_progress_draw'][f.result];
 if(f.result===4 ? !['simultaneous_exhaustion','no_legal_actions','ply_cap'].includes(f.end_reason)
                : f.end_reason!==expected)throw Error('终局原因与结果不一致');
}
function validReviewFrame(frame,enabled){
 if(!enabled && frame.move_origins===undefined && frame.flag_routes===undefined)return;
 if(!enabled || !Array.isArray(frame.move_origins) || frame.move_origins.length!==4 ||
    frame.move_origins.some(count=>!Number.isInteger(count)||count<0||count>25) ||
    !Array.isArray(frame.flag_routes) || frame.flag_routes.some(route=>!Array.isArray(route)||route.length!==3 ||
     route.some(v=>!Number.isInteger(v)) || route[0]<0 || route[0]>3 || route[1]<0 || route[1]>=129 ||
     route[2]<0 || route[2]>=129 || frame.seats[route[1]]!==route[0] || frame.truth[route[2]]!==0))
  throw Error('复盘明细字段错误');
}
function validFrame(f,i,single,detailed,review){
 if(f.decision){const d=f.decision;if(!['random-1','heuristic-1','heuristic-2','heuristic-3','flag-route-1'].includes(d.policy)||d.based_on_seq!==i-1||d.viewer!==f.event?.[0]||['objective','position','exposure','total'].some(k=>!Number.isInteger(d[k]))||d.total!==d.objective+d.position+d.exposure)throw Error('决策记录错误');}
 if(f.type!=='frame'||f.seq!==i||!Number.isInteger(f.turn)||f.turn<0||f.turn>3||!Number.isInteger(f.result)||f.result<0||f.result>4)throw Error('事件序号或状态错误');
 for(const key of ['seats','truth'])if(!Array.isArray(f[key])||f[key].length!==129)throw Error('棋盘数据不完整');
 if(!Array.isArray(f.views)||f.views.length!==4||f.views.some(v=>!Array.isArray(v)||v.length!==129))throw Error('观察记录不完整');
 for(let n=0;n<129;n++){
  if(!Number.isInteger(f.seats[n])||f.seats[n]<-1||f.seats[n]>3||!Number.isInteger(f.truth[n])||f.truth[n]<-1||f.truth[n]>11)throw Error('棋子数据越界');
  for(let v=0;v<4;v++)if(!Number.isInteger(f.views[v][n])||f.views[v][n]<-1||f.views[v][n]>11)throw Error('观察身份越界');
 }
 if(!Array.isArray(f.out)||f.out.length!==4)throw Error('出局记录错误');
 if(!Array.isArray(f.legal_move_counts)||f.legal_move_counts.length!==4||
    f.legal_move_counts.some(n=>!Number.isInteger(n)||n<0))throw Error('合法着计数错误');
 if(i&&(!Array.isArray(f.event)||f.event.length!==6||f.event.some(v=>!Number.isInteger(v))))throw Error('走法记录错误');
 if(i&&(f.event[0]<0||f.event[0]>3||f.event[1]<0||f.event[1]>=129||f.event[2]<0||f.event[2]>=129||
       f.event[3]<0||f.event[3]>3||f.event[4]<0||f.event[4]>15||f.event[5]<0||f.event[5]>15))throw Error('走法记录越界');
 validReviewFrame(f,review);
 if(i&&detailed)validSettlement(f,i);
 if(single&&(f.turn%2||f.seats.includes(1)||f.seats.includes(3)||!f.out[1]||!f.out[3]||
    (i&&![0,2].includes(f.event[0]))))throw Error('单挑记录出现未参赛座位');
}
function updateViewOptions(single){
 for(const side of ['east','west']){
  const option=$('view-'+side);option.disabled=single;option.hidden=single;
 }
 if(single&&[1,3].includes(Number($('view').value)))$('view').value='0';
}
function loggedMode(h){
 const ally=h.observation?.ally_visibility, enemy=h.observation?.enemy_identity_visible;
 if(ally==='none'&&enemy===false)return h.match==='single-dark'?'单挑暗棋':'四暗';
 if(h.match===undefined&&ally==='all'&&typeof enemy==='boolean')return enemy?'全明对局':'双明';
 throw Error('不支持此对局的观察规则');
}
function readExits(series,flagEliminates,detailed=false,exhaustion=false){
 const result=[];
 for(let i=1;i<series.length;i++){
  if(detailed){
   for(const exit of series[i].eliminations)result[exit.seat]={ply:exit.ply,reason:exitNames[exit.reason]};
   continue;
  }
  const [, ,to,outcome,mask]=series[i].event;
  if(!mask)continue;
  const before=series[i-1];
  const captured=flagEliminates&&(outcome===1||outcome===3)&&before.truth[to]===0?before.seats[to]:-1;
  for(let seat=0;seat<4;seat++){
   if((mask&(1<<seat))&&!result[seat])
    result[seat]={ply:i,reason:seat===captured?'军旗被夺':(exhaustion?'可动子耗尽':'无合法着')};
  }
 }
 return result;
}
function readArchiveRows(h,series){
 if(h.source_steps===undefined&&h.source_controls===undefined)return null;
 const rows=h.source_steps, controls=h.source_controls;
 if(h.review_version!==1||typeof h.source_tape!=='string'||!Array.isArray(rows)||rows.length!==series.length||rows[0]!==0||
    rows.some((step,i)=>!Number.isInteger(step)||(i>0&&step<=rows[i-1]))||!Array.isArray(controls))throw Error('真人原谱步号错误');
 const grouped=rows.map(step=>({step,events:[]}));
 for(const control of controls){
  if(!Array.isArray(control)||control.length!==5||control.some(n=>!Number.isInteger(n)))throw Error('真人原谱事件错误');
  const [ply,seq,kind,seat,cause]=control, row=grouped[ply];
  if(!row||ply<0||(ply===0&&kind!==2)||seq!==row.step+row.events.length+1||seq>=(rows[ply+1]??Infinity)||cause<0||cause>255||
     ([2,3,4].includes(kind) ? seat<0||seat>3 : kind!==5||seat!==-1||ply!==series.length-1))
   throw Error('真人原谱事件错误');
  row.events.push({seq,kind,seat,cause});
 }
 for(let i=1;i<grouped.length;i++)if(grouped[i].step!==grouped[i-1].step+grouped[i-1].events.length+1)
  throw Error('真人原谱事件序号有缺口');
 for(let i=1;i<grouped.length;i++){
  const mask=grouped[i].events.reduce((value,{kind,seat})=>(kind===3||kind===4)&&!series[i-1].out[seat]?value|(1<<seat):value,0);
  if(mask!==series[i].event[4])throw Error('真人原谱退场不完整');
 }
 const terminal=grouped.at(-1).events.at(-1);
 if(terminal?.kind!==5)throw Error('缺少真人原谱终局');
 if((terminal.cause===2)!==(series.at(-1).result===3))throw Error('真人原谱和棋结果与终局记录不一致');
 return grouped;
}
function archiveRetirementLabel({kind,cause}){
 if(kind===4)return '退出';
 return {1:'夺旗',2:'无合法着',3:'超时',4:'投降'}[cause]??`战败（原因码 ${cause}）`;
}
function archiveDrawLabel(frame, rules){
 const limit=rules['draw.no_progress.limit'];
 const count=frame.no_progress;
 const note=limit>0&&count===limit?'达到档案静步阈值':'原谱未记载和棋触发原因';
 return `原谱和棋 · 静步计数 ${count} · ${note}`;
}
function archiveEventLabel(event){
 const {kind,seat,cause}=event;
 if(kind===5)return `结束（原谱结果码 ${cause}）`;
 if(kind===2)return `${names[seat]} 跳过记录（次数 ${cause}）`;
 if(kind===4)return `${names[seat]} 退出记录`;
 return `${names[seat]} ${archiveRetirementLabel(event)}退场`;
}
function ruleCaption(rules, archived){
 const reveal=rules['marshal_death.reveal_flag']?'司令阵亡亮旗':'司令阵亡不亮旗';
 if(archived)return `原谱复盘 · ${reveal} · 跳过、退场和终局按原谱记录。`;
 const exhaustion=rules['elim.no_legal_move']==='eliminated_no_movable_piece';
 const retirement=exhaustion?'无着顺延，可动子耗尽出局':'按无合法着规则判退场';
 const limit=rules['draw.no_progress.limit'];
 const counter=rules['draw.no_progress.reset_on']==='capture'?'未吃子':'无碰撞';
 const draw=limit>0?`连续 ${limit} 手${counter}判和`:'无静步判和';
 return `本局规则 · ${reveal} · ${retirement} · ${draw}。`;
}
function load(text){
 stop();if(text.length>64000000)throw Error('文件超过64MB');if(!text.endsWith('\n'))throw Error('日志尾部不完整');
 const lines=text.trimEnd().split(/\r?\n/).map(x=>JSON.parse(x)); const h=lines.shift();
 if(!h||h.type!=='header'||h.schema!==1||h.engine!=='junqi-mvp-1'||h.topology!=='experimental-129-v1'||h.info_level!=='referee_private'||h.history_complete!==true)throw Error('不支持此格式或历史不完整');
 if(h.match!==undefined&&h.match!=='single-dark')throw Error('不支持此对局类型');
 const single=h.match==='single-dark';
 if(single&&(h.rules?.players!==2||JSON.stringify(h.rules?.teams)!=='[[0],[2]]'))throw Error('单挑阵营错误');
 loggedMode(h);
 if(!Array.isArray(h.nodes)||h.nodes.length!==129||h.nodes.some(p=>!Array.isArray(p)||p.length!==3||p.some(v=>!Number.isInteger(v))||p[0]<0||p[0]>16||p[1]<0||p[1]>16))throw Error('棋盘坐标错误');
 for(const k of ['roads','rails'])if(!Array.isArray(h[k])||h[k].some(e=>!Array.isArray(e)||e.length!==2||e.some(n=>!Number.isInteger(n)||n<0||n>=129)))throw Error('棋盘线路错误');
 const experimental=h.rules.experimental,detailed=h.settlement_detail===true;
 if(h.review_version!==undefined&&h.review_version!==1)throw Error('不支持此复盘明细版本');
 const exhaustion=experimental['elim.no_legal_move']==='eliminated_no_movable_piece';
 if(h.settlement_detail!==undefined&&(!detailed||!exhaustion))throw Error('自训结算标记错误');
 if(lines.length<2)throw Error('缺少对局记录');lines.forEach((frame,i)=>validFrame(frame,i,single,detailed,h.review_version===1));
 if(lines.at(-1).result===0)throw Error('对局未结束；请用完整日志');
 sourceRows=readArchiveRows(h,lines);
 header=h;frames=lines;exits=readExits(lines,experimental['elim.flag_removed_eliminates'],detailed,exhaustion);
 if(sourceRows)for(const [ply,row] of sourceRows.entries())for(const event of row.events){
  const explicit=row.events.some(other=>other.kind===3&&other.seat===event.seat);
  if(event.kind===3||event.kind===4&&!explicit&&exits[event.seat]?.ply===ply)
   exits[event.seat]={ply,reason:archiveRetirementLabel(event)};
 }
 index=0;actionReview=-1;reviewData=reviewSeries(lines,single);$('wholeGame').checked=true;$('replayInsights').hidden=false;
 updateViewOptions(single);$('error').hidden=true;$('timeline').max=frames.length-1;controls();render();
}
function svg(tag,attrs,parent=$('board')){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);parent.appendChild(e);return e;}
function renderBoard(f,view){
 const board=$('board');board.replaceChildren();
 for(const [kind,color,width]of [['roads','#3c4c40',.045],['rails','#7d8973',.10]])for(const [a,b]of header[kind]){
  const p=header.nodes[a],q=header.nodes[b];svg('line',{x1:p[0],y1:p[1],x2:q[0],y2:q[1],stroke:color,'stroke-width':width});
 }
 header.nodes.forEach(([x,y,k],n)=>{
  svg(k===1?'ellipse':'rect',k===1?{cx:x,cy:y,rx:.32,ry:.23,fill:'#263e30',stroke:'#55705a','stroke-width':.035}:{x:x-.13,y:y-.10,width:.26,height:.20,rx:.03,fill:k===2?'#706044':'#34433b'});
  if(f.seats[n]<0)return;const seat=f.seats[n],rank=view===4?f.truth[n]:f.views[view][n];
  svg('rect',{x:x-.42,y:y-.28,width:.84,height:.56,rx:.09,fill:rank<0?'#26322f':'#27392f',stroke:colors[seat],'stroke-width':.055});
  const t=svg('text',{x,y:y+.09,'text-anchor':'middle','font-size':rank<0?.23:.26,fill:colors[seat],'font-family':'Microsoft YaHei,sans-serif','font-weight':'600'});t.textContent=rank<0?'暗':ranks[rank];
 });
 if(f.event){const a=header.nodes[f.event[1]],b=header.nodes[f.event[2]];svg('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:'#f4e3ac','stroke-width':.075,'stroke-dasharray':'.13 .1','pointer-events':'none'});svg('circle',{cx:b[0],cy:b[1],r:.48,fill:'none',stroke:'#f4e3ac','stroke-width':.05});}
}
function render(){
 if(!frames.length)return;const f=frames[index],v=Number($('view').value),single=header.match==='single-dark';renderBoard(f,v);
 const mode=loggedMode(header);
 $('mode').textContent=v===4?'研究全明 · '+mode:names[v]+'视角 · '+mode;$('position').textContent=`第 ${index} / ${frames.length-1} 手`;
 const source=sourceRows?.[index];
 if(source)$('position').textContent+=` · 原谱 ${source.step}${source.events.length?'–'+source.events.at(-1).seq:''}`;
 $('privacy').textContent=v===4?'研究视角显示全部身份。':{
  '四暗':'队友与对手身份隐藏，公开揭示除外。',
  '单挑暗棋':'对手身份隐藏，公开揭示除外。',
  '双明':'己方与对家身份可见，敌方公开揭示除外。',
  '全明对局':'棋手在落子时可见全部身份。'
 }[mode];
 $('timeline').value=index;$('seed').textContent=source?'—':header.seed;$('turn').textContent=f.result?'—':names[f.turn];$('quiet').textContent=f.no_progress;
 const detail=header.settlement_detail;
 $('result').textContent=(detail?endNames[f.end_reason]:undefined)??['对局进行中',single?'南方获胜':'南北队获胜',single?'北方获胜':'东西队获胜','和棋','模拟截断（非和棋）'][f.result];
 if(source&&f.result===3)$('result').textContent=archiveDrawLabel(f,header.rules.experimental);
 if(source&&f.result===4)$('result').textContent='原谱结束，胜负待解码';
 const policy=header.policy??'random';
 const heuristicVersion=(typeof policy==='string'&&/^heuristic-([23])-/.exec(policy)?.[1])||1;
 $('policy').textContent=source?'真人原谱':({
  random:'随机合法走法（规则验证）','heuristic-all':`全席启发式 ${heuristicVersion}`,'flag-route':'夺旗练习棋手',
  'heuristic-ns':single?`南北均为启发式 ${heuristicVersion}`:`南北启发式 ${heuristicVersion}，东西随机`,
  'heuristic-ew':`东西启发式 ${heuristicVersion}，南北随机`,
  'heuristic-south':single?`南方启发式 ${heuristicVersion}，北方随机`:`南方启发式 ${heuristicVersion}，其余随机`,
  'heuristic-east':`东方启发式 ${heuristicVersion}，其余随机`
 }[heuristicVersion===1?policy:policy.replace(/^heuristic-[23]-/,'heuristic-')]??'未知棋手');
 $('teams').textContent=names.map((s,i)=>s+(single&&i%2?' 无参赛者':
  f.out[i]?' '+(exits[i]?.ply<=index?exits[i].reason+'出局':'已出局'):' 在场')).join(' · ');
 $('event').textContent=f.event?`${names[f.event[0]]}：节点 ${f.event[1]} → ${f.event[2]}。${['移动','攻击方胜','防守方胜','双方移除'][f.event[3]]}。`:
 (single?'南北各布 25 子，东西空翼照常通行。':source?'真人原谱的初始布局。':'初始布局。四个座位按各自观察选择合法走法。');
 const revealed=f.event?names.filter((_,i)=>f.event[5]&(1<<i)):[];
 if(revealed.length)$('event').textContent+=' '+revealed.join('、')+'司令阵亡，触发亮旗。';
 const currentExits=names.flatMap((s,i)=>exits[i]?.ply===index?[`${s}${exits[i].reason}出局。`]:[]);
 if(currentExits.length&&!source)$('event').textContent+=' '+currentExits.join(' ');
 if(source?.events.length)$('event').textContent+=' '+source.events.map(event=>
  `原谱 ${event.seq} ${archiveEventLabel(event)}`).join(' · ');
 const skipped=detail?names.filter((_,i)=>f.auto_skipped_mask&(1<<i)):[];
 if(skipped.length)$('event').textContent+=' '+skipped.join('、')+'暂时堵塞，自动顺延。';
 const d=f.decision;
 $('decisionView').disabled=!f.event;
 const allowed=d&&(v===4||v===d.viewer||mode==='全明对局'||(mode==='双明'&&(v%2===d.viewer%2)));
 $('decision').textContent=allowed ? `${d.policy} · 基于第 ${d.based_on_seq} 手的${names[d.viewer]}观察。目标 ${d.objective}，位置 ${d.position}，未知交战 ${d.exposure}，合计 ${d.total}。仅为启发式偏好分。`
  : d ? '切到行动者视角可查看该席记录。' : '这手没有启发式评分记录。';
 const limit=$('wholeGame').checked ? frames.length-1 : index;
 drawReviewChart(reviewData,index,limit,single);
 renderResources(f,v,single);
 renderExchange(reviewData,index,limit,v,frames);
 $('returnAction').hidden=actionReview<0||index>=actionReview;
 $('validation').textContent=ruleCaption(header.rules.experimental,Boolean(sourceRows))+' 页面已检查格式与序号，规则级校验由 CLI replay 完成。';
}
function move(n){stop();index=Math.max(0,Math.min(frames.length-1,n));render();}
$('first').onclick=()=>move(0);$('last').onclick=()=>move(frames.length-1);$('prev').onclick=()=>move(index-1);$('next').onclick=()=>move(index+1);
$('timeline').oninput=e=>move(Number(e.target.value));$('view').onchange=render;
$('play').onclick=()=>{if(timer){stop();return;}if(!frames.length)return;if(index===frames.length-1)index=0;$('play').textContent='暂停';timer=setInterval(()=>{index++;render();if(index>=frames.length-1)stop();},Number($('speed').value));};
$('speed').onchange=()=>stop();$('file').onchange=async e=>{try{const f=e.target.files[0];if(!f)return;if(f.size>64000000)throw Error('文件超过64MB');load(await f.text());}catch(err){fail(err);}};
$('wholeGame').onchange=render;
$('decisionView').onclick=()=>{
 if(!frames[index]?.event)return;
 actionReview=index;$('wholeGame').checked=false;$('view').value=String(frames[index].event[0]);move(index-1);
};
$('returnAction').onclick=()=>{if(actionReview<0)return;const step=actionReview;actionReview=-1;move(step);};
controls();if(window.JUNQI_LOG)try{load(window.JUNQI_LOG);}catch(e){fail(e);}
