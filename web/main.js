var frameViewer = document.getElementById('frameViewer');
var labelViewer = document.getElementById('labelViewer');

var video_source = document.getElementById('video_here');

var input_video = document.getElementById('input_video');
var input_trace = document.getElementById('input_trace');
var video_tag = document.getElementById('video_tag');

//var url=URL.createObjectURL(input_video.files[0])

var current_trace = null;

input_video.onchange = function(){
    video_source.src=URL.createObjectURL(input_video.files[0]);
    video_tag.load();
}

input_trace.onchange = function(event){
    var reader = new FileReader();
    reader.onload = function(event){
        current_trace = JSON.parse(event.target.result);
        console.log("trace loaded")
    };
    reader.readAsText(event.target.files[0]);
}

function render(){

}