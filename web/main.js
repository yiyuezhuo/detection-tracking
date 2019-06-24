var frameViewer = document.getElementById('frameViewer');
var labelViewer = document.getElementById('labelViewer');

var video_source = document.getElementById('video_here');

var input_video = document.getElementById('input_video');
var input_preprocess = document.getElementById('input_preprocess');
var video_tag = document.getElementById('video_tag');

//var url=URL.createObjectURL(input_video.files[0])

input_video.onchange = function(){
    video_source.src=URL.createObjectURL(input_video.files[0]);
    video_tag.load();
}

function render(){

}