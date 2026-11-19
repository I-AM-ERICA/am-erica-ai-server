'use strict';
const config = JSON.parse(document.getElementById('studio-config').textContent);
const $ = (id) => document.getElementById(id);
let drafts = [], selected = 0, busy = false, uploads = 0;
let images = {product_image: null, logo: null, style_reference: null};
const storeId = 'local-studio';
const templateNames = {editorial:'시그니처', bold_sale:'위크엔드 딜', soft_frame:'작은 선물', split:'컬렉션', night:'프리미엄', market:'동네의 발견',type_stack:'타이포 팝',magazine:'매거진',spotlight:'스튜디오 글로우',diagonal:'다이내믹',floating_type:'플로팅 타입',big_focus:'빅 포커스'};
const swatches = {editorial:'#e8dfcf', bold_sale:'#f44835', soft_frame:'#f5d6c4', split:'#7e9fd4', night:'#344044', market:'#bdd0a4',type_stack:'#932a30',magazine:'#e8dfcf',spotlight:'#41695a',diagonal:'#245be6',floating_type:'#3468ae',big_focus:'#ad450f'};
$('mode').textContent = config.mode === 'demo' ? '체험 모드 · AI 호출 없음' : 'CLOVA Studio 연결';
if (config.mode === 'demo') {
  $('reference').disabled = true;
  $('reference').title = 'CLOVA 연결 모드에서 사용할 수 있습니다.';
}
function message(text, error=false) {
  if (error) { $('error').textContent=text; $('error').hidden=false; $('status').textContent=''; }
  else { $('status').textContent=text; $('error').hidden=true; }
}
function lock(value) {
  busy=value; document.body.classList.toggle('busy',value);
  $('fields').disabled=value || uploads>0;
  $('sample').disabled=value || uploads>0;
  $('generate').disabled=value || uploads>0;
  $('revise').disabled=value;
  $('font-apply').disabled=value;$('font-select').disabled=value;
  document.querySelectorAll('.template,.suggestions button,.draft-card').forEach(b=>b.disabled=value);
}
async function api(path, body) {
  const controller = new AbortController();
  const timer=setTimeout(()=>controller.abort(),90000);
  try {
    const response=await fetch(path,{method:body?'POST':'GET',headers:{'Content-Type':'application/json','X-Studio-Token':config.token},body:body?JSON.stringify(body):undefined,signal:controller.signal});
    const value=await response.json();
    if (!response.ok) throw new Error(typeof value.detail==='string'?value.detail:'입력 정보를 확인해 주세요.');
    return value;
  } catch(error) {
    if(error.name==='AbortError') throw new Error('요청 시간이 초과됐어요. 잠시 후 다시 시도해 주세요.');
    if(error instanceof TypeError) throw new Error('서버에 연결하지 못했어요. 실행 상태를 확인해 주세요.');
    throw error;
  } finally {clearTimeout(timer);}
}
async function readImage(file) {
  if(!file || !file.size || file.size>5*1024*1024) throw new Error('이미지는 5MB 이하로 올려 주세요.');
  if(!['image/png','image/jpeg','image/webp'].includes(file.type)) throw new Error('PNG, JPG, WEBP 이미지를 선택해 주세요.');
  const data=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=()=>reject(new Error('이미지를 읽지 못했어요.'));reader.readAsDataURL(file);});
  const img=new Image();img.src=data;await img.decode();
  if(img.naturalWidth*img.naturalHeight>20000000) throw new Error('이미지를 2천만 화소 이하로 줄여 주세요.');
  return {base64:data.split(',')[1],media_type:file.type};
}
function preview(){
  const image=images.product_image;
  $('product-preview').hidden=!image;$('upload-label').hidden=!!image;$('remove-image').hidden=!image;
  if(image) $('product-preview').src='data:'+image.media_type+';base64,'+image.base64;
  else $('product-preview').removeAttribute('src');
}
for(const [id,key] of [['product-image','product_image'],['logo','logo'],['reference','style_reference']]){
  $(id).addEventListener('change',async()=>{
    uploads++;lock(busy);images[key]=null;
    try {if($(id).files[0]) images[key]=await readImage($(id).files[0]); message('');}
    catch(error){message(error.message,true);$(id).value='';}
    finally {uploads--;preview();lock(busy);}
  });
}
$('remove-image').onclick=()=>{images.product_image=null;$('product-image').value='';preview();};
$('sample').onclick=async()=>{
  uploads++;lock(false);
  try{
    const response=await fetch('/studio/sample-product.png');
    if(!response.ok) throw new Error('예시 이미지를 불러오지 못했어요.');
    const file=new File([await response.blob()],'sample-product.png',{type:'image/png'});
    images.product_image=await readImage(file);
    $('store').value='오후의 베이커리';$('product').value='버터 소금빵';
    $('message').value='바삭한 첫입, 촉촉한 속.\n오늘의 작은 여유를 담았어요.';
    $('price').value='3500';$('original-price').value='4000';
    $('period').value='11.20 – 11.27';$('contact').value='성수동 · 매장 포장 가능';
    $('sample-note').hidden=false;preview();message('예시를 채웠어요. 포스터 3개 만들기를 눌러 보세요.');
  }catch(error){message(error.message,true);}
  finally{uploads--;lock(false);}
};
const number=(id)=>$(id).value===''?null:Number($(id).value);
$('brief-form').addEventListener('submit',async(event)=>{
  event.preventDefault();if(busy||uploads)return;
  const brief={store:{store_id:storeId,name:$('store').value.trim()},product:$('product').value.trim(),message:$('message').value.trim(),
    tone:$('tone').value,price:number('price'),original_price:number('original-price'),period:$('period').value.trim(),contact:$('contact').value.trim(),background_mode:$('background-mode').value,...images};
  lock(true);message(config.mode==='demo'?'체험용 시안 3개를 만들고 있어요.':'우리 가게에 어울리는 시안을 만들고 있어요.');
  try {
    const result=await api('/v1/posters/drafts',brief);
    drafts=result.drafts;selected=0;render();
    message('시안 3개가 완성됐어요. 마음에 드는 한 장을 골라 보세요.');
    $('drafts').scrollIntoView({behavior:'smooth',block:'nearest'});
  }catch(error){message(error.message,true);}
  finally{lock(false);}
});
function render(){
  $('empty').hidden=true;$('drafts').hidden=false;$('editor').hidden=false;
  $('drafts').replaceChildren();
  drafts.forEach((draft,index)=>{
    const card=document.createElement('button');card.type='button';card.className='draft-card';
    card.setAttribute('aria-pressed',String(index===selected));card.setAttribute('aria-label','시안 '+(index+1)+' '+templateNames[draft.design.template_id]+' 선택');
    const image=document.createElement('img');image.src=draft.image_url;image.alt=draft.caption.headline+' 포스터';
    image.onerror=()=>message('이미지를 불러오지 못했어요. 서버의 이미지 주소 설정을 확인해 주세요.',true);
    const meta=document.createElement('div');meta.className='draft-meta';
    const name=document.createElement('strong');name.textContent=templateNames[draft.design.template_id];
    const families={editorial:'상품 중심',market:'상품 중심',type_stack:'상품 중심',spotlight:'상품 중심',bold_sale:'가격 중심',split:'가격 중심',diagonal:'가격 중심',floating_type:'타이포·대비 중심',big_focus:'상품·크기감 중심',soft_frame:'여백·서체 중심',night:'여백·서체 중심',magazine:'여백·서체 중심'};
    const version=document.createElement('small');version.textContent=families[draft.design.template_id]+' · v'+draft.version;
    meta.append(name,version);card.append(image,meta);
    card.onclick=()=>{if(!busy){selected=index;render();}};
    $('drafts').append(card);
  });
  const draft=drafts[selected];
  if(!$('font-select').options.length){
    for(const font of config.fonts){const option=document.createElement('option');option.value=font.id;option.textContent=font.name;$('font-select').append(option);}
  }
  $('font-select').value=draft.design.typography || 'auto';fontPreview();
  const processing=draft.image_processing || {};
  $('image-status').textContent=processing.warning || ({removed:'상품 배경을 제거하고 부드러운 그림자를 더했어요.',transparent:'투명한 상품 이미지에 부드러운 그림자를 더했어요.',kept:'원본 사진의 배경을 유지했어요.'}[processing.background] || '');
  $('image-status').hidden=!$('image-status').textContent;
  $('selected-label').textContent='선택한 시안 '+(selected+1)+' · 버전 '+draft.version;
  $('download').href=draft.image_url+'?download=true';
  $('caption').textContent=[draft.caption.headline,draft.caption.body,draft.caption.hashtags.join(' ')].filter(Boolean).join('\n\n');
  $('templates').replaceChildren();
  for(const [id,name] of Object.entries(templateNames)){
    const button=document.createElement('button');button.type='button';button.className='template';button.setAttribute('aria-pressed',String(draft.design.template_id===id));
    const art=document.createElement('i');art.dataset.layout=id;art.style.setProperty('--swatch',swatches[id]);button.append(art,document.createTextNode(name));
    button.onclick=()=>revise(id+' 템플릿으로 바꿔줘');$('templates').append(button);
  }
}
async function revise(instruction){
  if(busy||!drafts.length||!instruction.trim())return;
  const current=drafts[selected];lock(true);message('선택한 시안을 다듬고 있어요.');
  try{
    const next=await api('/v1/posters/'+current.id+'/revisions',{store_id:storeId,instruction,expected_version:current.version});
    drafts[selected]=next;$('instruction').value='';render();message('수정한 시안을 새 버전으로 저장했어요.');
  }catch(error){message(error.message,true);}
  finally{lock(false);}
}
$('revise').onclick=()=>revise($('instruction').value);
function fontPreview(){
  const font=config.fonts.find(font=>font.id===$('font-select').value);
  if(!font)return;
  $('font-description').textContent=font.description;
  const previewId=font.id==='auto'&&drafts.length?config.templateFonts[drafts[selected].design.template_id]:font.id;
  $('font-preview').src='/studio/font-preview/'+encodeURIComponent(previewId)+'.png';
}
$('font-select').onchange=fontPreview;
$('font-apply').onclick=()=>revise('제목 서체를 '+$('font-select').value+'로 바꿔줘');
$('instruction').onkeydown=(event)=>{if(event.key==='Enter'){event.preventDefault();revise($('instruction').value);}};
document.querySelectorAll('[data-edit]').forEach(button=>button.onclick=()=>revise(button.dataset.edit));
$('copy-caption').onclick=async()=>{try{await navigator.clipboard.writeText($('caption').textContent);message('홍보 문구를 복사했어요.');}catch{message('복사 권한을 확인해 주세요.',true);}};
