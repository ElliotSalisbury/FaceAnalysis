var container, camera, controls, scene, renderer;
var currMesh;

function init3d() {
    scene = new THREE.Scene();

    container = document.getElementById( '3dcanvas' );

    renderer = new THREE.WebGLRenderer();
    renderer.setClearColor( 0x444444 );
    renderer.setPixelRatio( window.devicePixelRatio );
    renderer.setSize( container.clientWidth, container.clientHeight );


    container.appendChild( renderer.domElement );

	camera = new THREE.PerspectiveCamera( 60, container.clientWidth / container.clientHeight, 1, 2000 );
    camera.position.x = 100;
    camera.position.z = 200;

    
    controls = new THREE.OrbitControls( camera, renderer.domElement );
    controls.target = new THREE.Vector3(0,0,-50);
    controls.enableDamping = true;
    controls.dampingFactor = 1.25;
    controls.enableZoom = true;

    light = new THREE.AmbientLight( 0xcccccc );
    scene.add( light );

    window.addEventListener( 'resize', onWindowResize, false );
}

function onWindowResize() {
    camera.left = container.clientWidth / - 2;
    camera.right = container.clientWidth / 2;
    camera.top = container.clientHeight / 2;
    camera.bottom = container.clientHeight / - 2;

    camera.updateProjectionMatrix();

    renderer.setSize( container.clientWidth, container.clientHeight );
}

function animate() {
    requestAnimationFrame( animate );
    controls.update();
    render();
}

function render() {
    renderer.render( scene, camera );
}

function setMesh(verts, faces) {
    var geometry = new THREE.Geometry();

    for(var i=0; i<verts.length; i++) {
        var vert = verts[i];
        geometry.vertices.push(new THREE.Vector3( vert[0], vert[1], vert[2]));
    }
    for(var i=0; i<faces.length; i++) {
        var face = faces[i];
        geometry.faces.push(new THREE.Face3(face[0], face[1], face[2]));
    }

    geometry.computeFaceNormals();

    if (currMesh) {
        scene.remove(currMesh);
    }

    // var material = new THREE.MeshPhongMaterial( { map: THREE.ImageUtils.loadTexture(FACENAME+'.jpg') } );
    var material = new THREE.MeshNormalMaterial();
    currMesh = new THREE.Mesh( geometry, material );
    scene.add( currMesh );
}