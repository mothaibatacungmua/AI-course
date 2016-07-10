var React = require('react');
var ReactDOM = require('react-dom');
var _ = require('underscore');
var PriorityQueue = require('./priority_queue').PriorityQueue;

function randInt(min, max){
    return Math.ceil(Math.random()*(max - min) + min);
}

function randPerm(n){
    var k = new Array(n);

    for(var i=0; i < n; i++){
        k[i] = i;
    }

    var index = 0;
    var swap;

    for(i=0; i < n; i++){
        index = randInt(0, n-1);

        swap = k[index];
        k[index] = k[i]
        k[i] = swap
    }

    return k;
}

var IMAGE_DIR = 'images';
var imageSrcs= [
    {src:'lol.png',begin:{},current:{}},
    {src:'A.png',begin:{},current:{}},
    {src:'N.png',begin:{},current:{}},
    {src:'H.png',begin:{},current:{}},
    {src:'M.png',begin:{},current:{}},
    {src:'U.png',begin:{},current:{}},
    {src:'O.png',begin:{},current:{}},
    {src:'N.png',begin:{},current:{}},
    {src:'beauty.png',begin:{},current:{}},
    {src:'X.png',begin:{},current:{}},
    {src:'E.png',begin:{},current:{}},
    {src:'P.png',begin:{},current:{}},
    {src:'H.png',begin:{},current:{}},
    {src:'I.png',begin:{},current:{}},
    {src:'N.png',begin:{},current:{}},
    {src:'H.png',begin:{},current:{}}
];

var N = 16;
var IMG_HEIGHT = 64;
var IMG_WIDTH = 64;

function reset(){
    var squareN = Math.sqrt(N);

    for(var i=0; i < N; i++){
        imageSrcs[i].begin = {
            y: i%squareN,
            x: Math.floor(i/squareN)
            
        };

        imageSrcs[i].current = {
            y: i%squareN,
            x: Math.floor(i/squareN)
            
        };
    }
}

function shuffe(){
    var k = randPerm(N);
    var squareN = Math.sqrt(N);
    var ret;

    for(var i=0; i < N; i++){
        imageSrcs[i].current = {
            y: k[i] % squareN,
            x: Math.floor(k[i]/squareN)
            
        }

        if(imageSrcs[i].begin.x == 0 && imageSrcs[i].begin.y == 0){
            ret = imageSrcs[i];
        }
    }

    return ret.current;
}

function calcMathanDist(A, B){
    return Math.abs(A.x - B.x) + Math.abs(A.y - B.y);
}

function draw(ctx){
    
    var i;
    var squareN = Math.sqrt(N);
    
    function loadImages(callback){
        var loaded = 0;
        var keep = [];

        for(i=0; i < N; i++){
            var load = new Image();
            keep.push(load);
            load.onload = function (){
                if(++loaded == N) return callback(keep);

            }

            load.src = IMAGE_DIR + '/' + imageSrcs[i].src;
        }
    }

    //draw image
    loadImages(function(imgObjs){
        for(i=0; i < N; i++){
            ctx.drawImage(
                imgObjs[imageSrcs[i].current.x*squareN + imageSrcs[i].current.y], 
                (i%squareN)*(IMG_WIDTH+1)+1, 
                Math.floor(i/squareN)*(IMG_HEIGHT+1)+1);
        }
    });

    //draw grid
    for(i=0; i < (squareN + 1); i++){
        ctx.moveTo((IMG_WIDTH+1)*i, 0);
        ctx.lineTo((IMG_WIDTH+1)*i, (IMG_HEIGHT+1)*squareN);

        ctx.moveTo(0, (IMG_HEIGHT+1)*i);
        ctx.lineTo((IMG_WIDTH+1)*squareN, (IMG_HEIGHT+1)*i);
    }

    ctx.stroke();
}

function copyCell(ctx, pos){
    return ctx.getImageData(pos.y*(IMG_WIDTH+1)+1, pos.x*(IMG_HEIGHT+1)+1, IMG_WIDTH, IMG_HEIGHT);
}

function putCell(ctx, imageDat, pos){
    return ctx.putImageData(imageDat, pos.y*(IMG_WIDTH+1)+1, pos.x*(IMG_HEIGHT+1)+1);
}

function swapCell(ctx, pos1, pos2){
    var cell1 = copyCell(ctx, pos1);
    var cell2 = copyCell(ctx, pos2);

    putCell(ctx, cell2, pos1);
    putCell(ctx, cell1, pos2);
}

function moveLeft(ctx, current){
    if((current.x - 1) < 0) throw "Cannot move to left";

    var target = {x:current.x-1, y:current.y};
    swapCell(ctx, current, target);

    return target;
}

function moveRight(ctx, current){
    var squareN = Math.sqrt(N);

    if((current.x+1) > (squareN-1)) throw "Cannot move to right";

    var target = {x:current.x+1, y:current.y};
    swapCell(ctx, current, target);

    return target;
}

function moveUp(ctx, current){
    if((current.y - 1) < 0) throw "Cannot move to up";

    var target = {x:current.x, y:current.y-1}
    swapCell(ctx, current, target);

    return target;
}

function moveDown(ctx, current){
    var squareN = Math.sqrt(N);
    if((current.y + 1) > (squareN-1)) throw "Cannot move to down";

    var target = {x:current.x, y:current.y+1};
    swapCell(ctx, current, {x:current.x, y:current.y+1});

    return target;
}

/* calc h(n) */
function mathanHeur(mapping){
    var heur = 0;
    var squareN = Math.sqrt(N);

    for(var i=0;i < squareN; i++){
        for(var j=0;j < squareN; j++){
            if((mapping[i][j].x !== 0) || (mapping[i][j].y !== 0)){
                heur += calcMathanDist(mapping[i][j], {x:i, y:j});    
            }
            
        }
    }

    return heur;
}

function cloneArray(array){
    var n = array.length;
    var newArray = new Array(n);

    for(var i=0; i < n; i++){
        newArray[i] = new Array(array[i].length);
        for(var j=0; j < array[i].length; j++){
            newArray[i][j] = _.clone(array[i][j]);
        }
        
    }

    return newArray;
}

function checkEqualPos(posA, posB){
    return (posA.x === posB.x) && (posA.y === posB.y);
}

function checkEqualMap(mapA, mapB){
    for(var i=0; i < mapA.length; i++){
        for(var j=0; j < mapA[i].length; j++){
            if(!checkEqualPos(mapA[i][j], mapB[i][j])){
                return false;
            }
        }
    }

    return true;
}


function isVisisted(closed, state){
    if(closed.length === 0) return false;

    for(var i=0; i < closed.length; i++){
        if(checkEqualMap(closed[i], state.map)){
            //throw "isVisisted";
            return true;
        }
    }

    return false;
}

function createSuccessor(current, action, target, cost){
    var newMap = cloneArray(current.map);

    newMap[current.pos.x][current.pos.y] = _.clone(current.map[target.x][target.y]);
    newMap[target.x][target.y] = _.clone(current.map[current.pos.x][current.pos.y])

    return {action:action, pos:target, cost:cost, map:newMap};
}

function getSuccessors(current){
    var ret = [];
    var squareN = Math.sqrt(N);

    if((current.pos.x - 1) >= 0){
        ret.push(createSuccessor(current, 'left', {x:current.pos.x-1, y:current.pos.y}, 1));
    }

    if((current.pos.x + 1) < squareN){
        ret.push(createSuccessor(current, 'right', {x:current.pos.x+1, y:current.pos.y}, 1));
    }

    if((current.pos.y - 1) >= 0){
        ret.push(createSuccessor(current, 'up', {x:current.pos.x, y:current.pos.y-1}, 1));
    }

    if((current.pos.y + 1) < squareN){
        ret.push(createSuccessor(current, 'down', {x:current.pos.x, y:current.pos.y+1}, 1));
    }

    return ret;
}

function isGoalState(state){
    return (mathanHeur(state.map) === 0);
}

function clonePath(path){
    var length = path.length;
    var clone = [];

    for(var i=0 ; i < length;i++){
        clone.push({
            action:path[i].action, 
            pos:_.clone(path[i].pos), 
            cost:path[i].cost, 
            map:cloneArray(path[i].map)});
    }


    return clone;
}

function calcFn(A){
    //return (A.length + mathanHeur(A[A.length-1].map));
    return (mathanHeur(A[A.length-1].map));
}

function AStar(problem){
    var closed = [];
    var queue = new PriorityQueue(function (A, B){
        return A.fn - B.fn;
    });
    
    var obj;

    queue.push({path:[problem.startState], fn:calcFn([problem.startState])});

    var count = 0;
    while(queue.length > 0){
        count++;
        obj = queue.shift();
        var travel = obj.path;
        var lastNode = travel[travel.length-1];

        if(isGoalState(lastNode)){
            break;
        }


        if(count % 1000 === 0){
            console.log('travel_'+count+':',mathanHeur(lastNode.map), travel.length);
        }

        if(!isVisisted(closed, lastNode)){
            closed.push(lastNode.map);
            var successors = getSuccessors(lastNode);


            for(var i=0; i < successors.length; i++){
                //if(count === 1) {console.log(successors[i]); console.log(mathanHeur(successors[i].map)); break;}
                if(!isVisisted(closed, successors[i])){
                    var newTravel = clonePath(travel);
                    newTravel.push(successors[i]);
                    queue.push({path:newTravel, fn:calcFn(newTravel)});
                }
            }

            //if(count == 2) return 0;
        }

    }

    
    var actions = travel.map(function(item){return item.action;});
    console.log(actions);
    console.log('AStar Finished!, Expanded:'+ count + ', Steps:' + actions.length);
    return actions;
}

function reverseDirection(direction){
    if(direction == 'left') return 'right';

    if(direction == 'right') return 'left';

    if(direction == 'up') return 'down';

    if(direction == 'down') return 'up';
}

function RBFS(problem){
    var squareN = Math.sqrt(N);
    var steps = [];
    var count = 0;

    function search(node, depth, fLimit){
        count++;

        if(count % 100000 == 0){
            console.log('travel_'+ count +':',mathanHeur(node.map), depth, fLimit);
        }

        if(isGoalState(node)){
            steps.push(node);
            return {goal:true, step:node, fBest:0};
        }

        var successors = getSuccessors(node);        
        var s = [];
        for(var i=0; i < successors.length; i++){
            s.push(
            {
                f:Math.max(mathanHeur(node.map) + depth, mathanHeur(successors[i].map) +depth+1),
                //f:mathanHeur(successors[i].map),
                node:successors[i]
            }) ;
        }

        // if(count == 1) {
        //     console.log(s);
        //     return;
        // }

        while(true){
            s.sort(function (A, B){return A.f - B.f;});

            if(s[0].f > fLimit) return {goal:false, step:null, fBest: s[0].f};

            var alternative = s[1].f;

            var ret = search(s[0].node, depth+1, Math.min(fLimit, alternative));
            s[0].f = ret.fBest;

            if(ret.goal){
                steps.push(s[0].node);
                return {goal:true, step: s[0].node, fBest:0};
            }
        }
    }

    search(problem.startState, 0, 500);
    steps.splice(0,1);
    steps.reverse();

    console.log('RBFS Finished!, Expanded:'+count+', Setup:' +steps.length);
    var actions = steps.map(function(item){return item.action;});

    return actions;    
}

var testMap0 = [
    [{"x":0,"y":2},{"x":0,"y":1},{"x":3,"y":2},{"x":2,"y":1}],
    [{"x":0,"y":3},{"x":3,"y":0},{"x":1,"y":0},{"x":3,"y":3}],
    [{"x":2,"y":3},{"x":2,"y":0},{"x":0,"y":0},{"x":3,"y":1}],
    [{"x":1,"y":2},{"x":2,"y":2},{"x":1,"y":3},{"x":1,"y":1}]
];

var testMap1 = [
    [{"x":0,"y":1},{"x":0,"y":2},{"x":0,"y":3},{"x":1,"y":3}],
    [{"x":1,"y":0},{"x":1,"y":1},{"x":1,"y":2},{"x":0,"y":0}],
    [{"x":2,"y":0},{"x":2,"y":1},{"x":2,"y":2},{"x":2,"y":3}],
    [{"x":3,"y":0},{"x":3,"y":1},{"x":3,"y":2},{"x":3,"y":3}]
];

function map(){
    var squareN = Math.sqrt(N);
    var m = new Array(squareN);

    for(var i=0;i<squareN;i++){
        m[i] = new Array(squareN);
    }

    for(i = 0; i < N; i++){
        m[Math.floor(i/squareN)][i%squareN] = _.clone(imageSrcs[i].current);
    }

    return m;
}

function initWithMap(constMap){
    var squareN = Math.sqrt(N);
    var ret = {};

    for(var i=0;i < squareN;i++){       //loop y
        for(var j=0;j < squareN;j++){   //loop x
            imageSrcs[i*squareN + j].current = _.clone(constMap[i][j]);

            if(constMap[i][j].x == 0 && constMap[i][j].y == 0){
                ret.x = i;
                ret.y = j;
            }
        }
    }

    return ret;
}
var App = React.createClass({
    getInitialState: function () {
        return {};
    },
    run: function (){
        var actions = RBFS({startState:this.startState});
        var current = this.startPos;
        var i = 0;

        var clear = setInterval(function(){
            switch(actions[i++]){
                case 'left':
                    current = moveLeft(this.ctx, current);
                    break;
                case 'right':
                    current = moveRight(this.ctx, current);
                    break;
                case 'up':
                    current = moveUp(this.ctx, current);
                    break;
                case 'down':
                    current = moveDown(this.ctx, current);
                    break;
            }

            if(i >= actions.length){
                clearInterval(clear);    
            }
            
        }.bind(this),200);
    },
    componentDidMount: function () {
        var canvas = ReactDOM.findDOMNode(this.refs._canvas);
        canvas.width  = 270;
        canvas.height = 270;
        this.ctx = canvas.getContext('2d');
        reset();
        this.startPos = shuffe();
        //this.startPos = initWithMap(testMap0);
        this.startState = {action:null, pos:this.startPos, cost:0, map:map()};
        //this.startState = {action:null, pos:{x:0,y:0}, cost:0, map:map()};
        draw(this.ctx);
    },
    render: function (){
        return (
            <div>
                <div style={{"height":"100%"}}>
                    <canvas ref="_canvas"></canvas>
                </div>
                <div>
                    <button onClick={this.run}>Run</button>
                </div>
            </div>
        )
    }
});

ReactDOM.render(
    <App />,
    document.getElementById('app')
);