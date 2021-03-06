var frameViewer = document.getElementById('frameViewer');
var labelViewer = document.getElementById('labelViewer');

var video_source = document.getElementById('video_here');

var input_video = document.getElementById('input_video');
var input_trace = document.getElementById('input_trace');
var video_tag = document.getElementById('video_tag');

var curtain = document.getElementById('curtain');
var ctx = curtain.getContext('2d');

var percent = null;

//var url=URL.createObjectURL(input_video.files[0])

var current_trace = null;
var video_loaded = false;

input_video.onchange = function(){
    video_source.src=URL.createObjectURL(input_video.files[0]);
    video_tag.load();
    
    // setup
    video_tag.ontimeupdate = render;
    curtain.width = video_tag.offsetWidth;
    curtain.height = video_tag.offsetHeight;
    video_loaded = true;
}

input_trace.onchange = function(event){
    var reader = new FileReader();
    reader.onload = function(event){
        current_trace = JSON.parse(event.target.result);
        console.log("trace loaded");

        console.log(video_tag.offsetWidth, video_tag.offsetHeight,
            current_trace.shape[0], current_trace.shape[1],
            video_tag.offsetWidth/current_trace.shape[0], video_tag.offsetHeight/current_trace.shape[1])
        percent = video_tag.offsetWidth / current_trace.shape[0];
    };
    reader.readAsText(event.target.files[0]);
}

function rectangle(pt1, pt2, color, thickness){
    ctx.beginPath();
    ctx.lineWidth = thickness * percent;
    //ctx.strokeStyle = 'rgb('+color[0]+','+color[1]+','+color[2]+')'//color;
    ctx.strokeStyle = 'blue'
    ctx.rect(pt1[0] * percent, pt1[1] * percent, (pt2[0]-pt1[0]) * percent, (pt2[1]-pt1[1])* percent);
    ctx.stroke();    
}


function arrowedLine(pt1, pt2, color, thickness){
    //canvas_arrow(ctx, pt1[0]*percent, pt1[1]*percent, pt2[0]*percent, pt2[1]*percent)
    // It seems that sample don't have arrow so I don't need to draw them.
}

function putText(text, org, fontFace, fontScale, color, thickness, lineType){
    //ctx.font = fontScale/percent +'px serif';
    ctx.font = '20px serif';
    ctx.fillStyle = 'blue';
    ctx.fillText(text, org[0]*percent+50, org[1]*percent);
}

function appendItem(master, key, value){
    var key_el = document.createElement("label");
    key_el.textContent = key;
    //key_el.class = 'attr';
    key_el.setAttribute('class', 'attr');

    var value_el = document.createElement("input");
    //value_el.type = 'text';
    //value_el.value = value;
    //value_el.class = 'class';
    value_el.setAttribute('type', 'text');
    value_el.setAttribute('value', value);
    value_el.setAttribute('class', 'value');

    master.appendChild(key_el);
    master.appendChild(value_el);
}

direc_map = {
    'none': '无',
    'left': '左',
    'right': '右'
}

yes_no_map = {
    true: '是',
    false: '否'
}

function createLabel(idx, coord, width, height, direc, speed, loaded, hc){
    //document.createElement("LI"); 
    var master = document.createElement('div');

    var header = document.createElement("label");
    //header.class = 'header';
    header.setAttribute('class', 'header');
    header.textContent = '船只'+ idx;

    master.appendChild(header);

    appendItem(master, "长" , width);
    appendItem(master, "宽" , height);
    appendItem(master, "中心点" , coord);
    appendItem(master, "方向" , direc_map[direc]);
    appendItem(master, "速度" , speed);
    appendItem(master, "载货" , yes_no_map[loaded]);
    appendItem(master, "危化" , yes_no_map[hc]);

    return master
}

function render(event){
    if(! current_trace || ! video_loaded){
        return;
    }
    var currentTime = event.target.currentTime;
    var duration = event.target.duration;
    console.log(currentTime);
    var frame_idx = Math.floor(current_trace.frames.length * (currentTime/duration));
    var frame = current_trace.frames[frame_idx];

    ctx.clearRect(0, 0, curtain.width, curtain.height);
    for(var command of frame.command_list){
        switch(command[0]){
            case "rectangle":
                rectangle(command[1],command[2],command[3],command[4]);
                break;
            case "arrowedLine":
                arrowedLine(command[1],command[2],command[3],command[4]);
                break;
            case "putText":
                putText(command[1],command[2],command[3],command[4],command[5],command[6],command[7]);
                break;
        }
    }


    labelViewer.innerHTML = '';
    //for(legend of frame.legend_list){
        for(legend_idx in frame.legend_list){
        var legend = frame.legend_list[legend_idx];
        var [_, predict] = frame.predict_list[legend_idx]; // predict=0 => empty; 1=> hc; 2=>loaded
        var loaded = predict == 2;
        var hc = predict == 1;
        var [idx, coord, width, height, direc, speed] = legend;
        var label_box = createLabel(idx, coord, width, height, direc, speed, loaded, hc);
        labelViewer.appendChild(label_box);
    }
}

// DEBUG
var label_box = createLabel('1', '2', '3', '4', '5', '6');
labelViewer.appendChild(label_box);