// NanoGPT 3D Visualizer
class NanoGPTVisualizer {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.animationId = null;
        this.components = new Map();
        this.isAnimating = false;
        this.animationSpeed = 1;
        this.educationalMode = false;
        
        // Model configuration
        this.config = {
            embedding_dim: 768,
            num_heads: 12,
            num_layers: 6,
            vocab_size: 50257,
            context_length: 1024,
            ff_dim: 3072
        };
        
        this.init();
    }
    
    init() {
        this.setupScene();
        this.setupLighting();
        this.setupCamera();
        this.setupRenderer();
        this.setupControls();
        this.createComponents();
        this.setupEventListeners();
        this.animate();
        
        // Hide loading screen
        setTimeout(() => {
            document.getElementById('loading-screen').classList.add('hidden');
        }, 2000);
    }
    
    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a1a);
        this.scene.fog = new THREE.Fog(0x1a1a1a, 50, 200);
    }
    
    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);
        
        // Directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);
        
        // Point lights for accent
        const pointLight1 = new THREE.PointLight(0x32b8c6, 0.5, 100);
        pointLight1.position.set(-20, 10, 20);
        this.scene.add(pointLight1);
        
        const pointLight2 = new THREE.PointLight(0x21808d, 0.3, 100);
        pointLight2.position.set(20, 10, -20);
        this.scene.add(pointLight2);
    }
    
    setupCamera() {
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        this.camera.position.set(0, 15, 25);
        this.camera.lookAt(0, 0, 0);
    }
    
    setupRenderer() {
        const canvas = document.getElementById('three-canvas');
        this.renderer = new THREE.WebGLRenderer({ 
            canvas: canvas,
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    }
    
    setupControls() {
        // Basic orbit controls using mouse events
        this.controls = {
            enabled: true,
            mouseX: 0,
            mouseY: 0,
            targetX: 0,
            targetY: 0,
            distance: 25,
            targetDistance: 25
        };
        
        const canvas = this.renderer.domElement;
        let isDragging = false;
        let isPanning = false;
        let previousMousePosition = { x: 0, y: 0 };
        
        canvas.addEventListener('mousedown', (event) => {
            if (event.button === 0) {
                isDragging = true;
            } else if (event.button === 2) {
                isPanning = true;
            }
            previousMousePosition = { x: event.clientX, y: event.clientY };
        });
        
        canvas.addEventListener('mousemove', (event) => {
            if (isDragging) {
                const deltaX = event.clientX - previousMousePosition.x;
                const deltaY = event.clientY - previousMousePosition.y;
                
                this.controls.targetX += deltaX * 0.01;
                this.controls.targetY += deltaY * 0.01;
            } else if (isPanning) {
                const deltaX = event.clientX - previousMousePosition.x;
                const deltaY = event.clientY - previousMousePosition.y;
                
                this.camera.position.x -= deltaX * 0.05;
                this.camera.position.y += deltaY * 0.05;
            }
            previousMousePosition = { x: event.clientX, y: event.clientY };
        });
        
        canvas.addEventListener('mouseup', () => {
            isDragging = false;
            isPanning = false;
        });
        
        canvas.addEventListener('wheel', (event) => {
            this.controls.targetDistance += event.deltaY * 0.01;
            this.controls.targetDistance = Math.max(5, Math.min(100, this.controls.targetDistance));
        });
        
        canvas.addEventListener('contextmenu', (event) => {
            event.preventDefault();
        });
    }
    
    createComponents() {
        this.createTokenEmbedding();
        this.createPositionalEmbedding();
        this.createTransformerBlocks();
        this.createOutputLayer();
        this.createConnections();
    }
    
    createTokenEmbedding() {
        const group = new THREE.Group();
        
        // Create embedding matrix representation
        const embedGeometry = new THREE.BoxGeometry(8, 1, 6);
        const embedMaterial = new THREE.MeshLambertMaterial({ 
            color: 0x1fb8cd,
            transparent: true,
            opacity: 0.8
        });
        
        const embedMesh = new THREE.Mesh(embedGeometry, embedMaterial);
        embedMesh.position.set(0, -15, 0);
        group.add(embedMesh);
        
        // Add smaller cubes to represent tokens
        for (let i = 0; i < 20; i++) {
            const tokenGeometry = new THREE.BoxGeometry(0.3, 0.3, 0.3);
            const tokenMaterial = new THREE.MeshLambertMaterial({ 
                color: 0xffc185 
            });
            const tokenMesh = new THREE.Mesh(tokenGeometry, tokenMaterial);
            tokenMesh.position.set(
                (Math.random() - 0.5) * 10,
                -13,
                (Math.random() - 0.5) * 8
            );
            group.add(tokenMesh);
        }
        
        group.userData = {
            name: 'Token Embedding',
            type: 'embedding',
            description: 'Converts input tokens to dense vector representations',
            parameters: '38.6M'
        };
        
        this.components.set('token-embedding', group);
        this.scene.add(group);
    }
    
    createPositionalEmbedding() {
        const group = new THREE.Group();
        
        // Create sine wave representation
        const curve = new THREE.EllipseCurve(
            0, 0,
            6, 3,
            0, 2 * Math.PI,
            false,
            0
        );
        
        const points = curve.getPoints(50);
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({ 
            color: 0xb4413c,
            linewidth: 3
        });
        
        const line = new THREE.Line(geometry, material);
        line.position.set(0, -12, 0);
        line.rotation.x = Math.PI / 2;
        group.add(line);
        
        // Add position markers
        for (let i = 0; i < 10; i++) {
            const markerGeometry = new THREE.SphereGeometry(0.1, 8, 8);
            const markerMaterial = new THREE.MeshLambertMaterial({ 
                color: 0xecebd5 
            });
            const marker = new THREE.Mesh(markerGeometry, markerMaterial);
            marker.position.set(
                Math.cos(i * 0.6) * 6,
                -12,
                Math.sin(i * 0.6) * 3
            );
            group.add(marker);
        }
        
        group.userData = {
            name: 'Positional Embedding',
            type: 'embedding',
            description: 'Adds position information to token embeddings',
            parameters: '0.8M'
        };
        
        this.components.set('positional-embedding', group);
        this.scene.add(group);
    }
    
    createTransformerBlocks() {
        for (let layer = 0; layer < this.config.num_layers; layer++) {
            const yPos = -8 + (layer * 3);
            
            // Multi-head attention
            this.createMultiHeadAttention(layer, yPos);
            
            // Layer normalization
            this.createLayerNorm(layer, yPos + 0.5);
            
            // Feed-forward network
            this.createFeedForward(layer, yPos + 1);
            
            // Residual connections
            this.createResidualConnections(layer, yPos);
        }
    }
    
    createMultiHeadAttention(layer, yPos) {
        const group = new THREE.Group();
        
        // Create attention heads
        for (let head = 0; head < this.config.num_heads; head++) {
            const headGroup = new THREE.Group();
            
            // Q, K, V matrices
            const qGeometry = new THREE.BoxGeometry(0.5, 0.5, 0.5);
            const kGeometry = new THREE.BoxGeometry(0.5, 0.5, 0.5);
            const vGeometry = new THREE.BoxGeometry(0.5, 0.5, 0.5);
            
            const qMaterial = new THREE.MeshLambertMaterial({ color: 0x5d878f });
            const kMaterial = new THREE.MeshLambertMaterial({ color: 0xdb4545 });
            const vMaterial = new THREE.MeshLambertMaterial({ color: 0xd2ba4c });
            
            const qMesh = new THREE.Mesh(qGeometry, qMaterial);
            const kMesh = new THREE.Mesh(kGeometry, kMaterial);
            const vMesh = new THREE.Mesh(vGeometry, vMaterial);
            
            const angle = (head / this.config.num_heads) * Math.PI * 2;
            const radius = 4;
            
            qMesh.position.set(
                Math.cos(angle) * radius,
                yPos,
                Math.sin(angle) * radius
            );
            kMesh.position.set(
                Math.cos(angle + 0.5) * radius,
                yPos,
                Math.sin(angle + 0.5) * radius
            );
            vMesh.position.set(
                Math.cos(angle + 1) * radius,
                yPos,
                Math.sin(angle + 1) * radius
            );
            
            headGroup.add(qMesh);
            headGroup.add(kMesh);
            headGroup.add(vMesh);
            
            // Attention connections
            const connectionGeometry = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(0, 0, 0),
                qMesh.position,
                kMesh.position,
                vMesh.position
            ]);
            const connectionMaterial = new THREE.LineBasicMaterial({ 
                color: 0x964325,
                transparent: true,
                opacity: 0.3
            });
            const connectionLine = new THREE.Line(connectionGeometry, connectionMaterial);
            headGroup.add(connectionLine);
            
            group.add(headGroup);
        }
        
        group.userData = {
            name: `Multi-Head Attention Layer ${layer + 1}`,
            type: 'attention',
            description: 'Parallel attention mechanisms capturing different relationships',
            layer: layer
        };
        
        this.components.set(`attention-${layer}`, group);
        this.scene.add(group);
    }
    
    createLayerNorm(layer, yPos) {
        const group = new THREE.Group();
        
        // Create normalization representation
        const normGeometry = new THREE.CylinderGeometry(2, 2, 0.2, 16);
        const normMaterial = new THREE.MeshLambertMaterial({ 
            color: 0x944454,
            transparent: true,
            opacity: 0.7
        });
        
        const normMesh = new THREE.Mesh(normGeometry, normMaterial);
        normMesh.position.set(0, yPos, 0);
        group.add(normMesh);
        
        group.userData = {
            name: `Layer Normalization ${layer + 1}`,
            type: 'normalization',
            description: 'Stabilizes training and improves convergence',
            layer: layer
        };
        
        this.components.set(`layernorm-${layer}`, group);
        this.scene.add(group);
    }
    
    createFeedForward(layer, yPos) {
        const group = new THREE.Group();
        
        // Create MLP representation
        const mlpGeometry = new THREE.BoxGeometry(3, 1, 2);
        const mlpMaterial = new THREE.MeshLambertMaterial({ 
            color: 0x13343b,
            transparent: true,
            opacity: 0.8
        });
        
        const mlpMesh = new THREE.Mesh(mlpGeometry, mlpMaterial);
        mlpMesh.position.set(0, yPos, 0);
        group.add(mlpMesh);
        
        // Add expanding/contracting visual
        const expandGeometry = new THREE.BoxGeometry(4, 0.5, 2.5);
        const expandMaterial = new THREE.MeshLambertMaterial({ 
            color: 0x1fb8cd,
            transparent: true,
            opacity: 0.5
        });
        
        const expandMesh = new THREE.Mesh(expandGeometry, expandMaterial);
        expandMesh.position.set(0, yPos + 0.3, 0);
        group.add(expandMesh);
        
        group.userData = {
            name: `Feed Forward Layer ${layer + 1}`,
            type: 'mlp',
            description: 'Two-layer neural network with GELU activation',
            layer: layer
        };
        
        this.components.set(`feedforward-${layer}`, group);
        this.scene.add(group);
    }
    
    createResidualConnections(layer, yPos) {
        const group = new THREE.Group();
        
        // Create residual connection lines
        const points = [
            new THREE.Vector3(-5, yPos - 0.5, 0),
            new THREE.Vector3(-6, yPos, 0),
            new THREE.Vector3(-6, yPos + 1.5, 0),
            new THREE.Vector3(-5, yPos + 2, 0)
        ];
        
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({ 
            color: 0xffc185,
            linewidth: 2
        });
        
        const line = new THREE.Line(geometry, material);
        group.add(line);
        
        group.userData = {
            name: `Residual Connection ${layer + 1}`,
            type: 'connection',
            description: 'Skip connection for gradient flow',
            layer: layer
        };
        
        this.components.set(`residual-${layer}`, group);
        this.scene.add(group);
    }
    
    createOutputLayer() {
        const group = new THREE.Group();
        
        // Create output projection
        const outputGeometry = new THREE.BoxGeometry(10, 2, 4);
        const outputMaterial = new THREE.MeshLambertMaterial({ 
            color: 0xb4413c,
            transparent: true,
            opacity: 0.8
        });
        
        const outputMesh = new THREE.Mesh(outputGeometry, outputMaterial);
        outputMesh.position.set(0, 12, 0);
        group.add(outputMesh);
        
        // Softmax representation
        for (let i = 0; i < 20; i++) {
            const barGeometry = new THREE.BoxGeometry(0.3, Math.random() * 2 + 0.5, 0.3);
            const barMaterial = new THREE.MeshLambertMaterial({ 
                color: 0xecebd5 
            });
            const barMesh = new THREE.Mesh(barGeometry, barMaterial);
            barMesh.position.set(
                (i - 10) * 0.4,
                14,
                0
            );
            group.add(barMesh);
        }
        
        group.userData = {
            name: 'Output Layer',
            type: 'output',
            description: 'Final projection to vocabulary size for token prediction',
            parameters: '38.6M'
        };
        
        this.components.set('output-layer', group);
        this.scene.add(group);
    }
    
    createConnections() {
        // Create data flow connections between major components
        const connectionPoints = [
            { from: [0, -15, 0], to: [0, -12, 0] }, // Token to positional
            { from: [0, -12, 0], to: [0, -8, 0] },  // Positional to first layer
            { from: [0, 7, 0], to: [0, 12, 0] }     // Last layer to output
        ];
        
        connectionPoints.forEach((connection, index) => {
            const points = [
                new THREE.Vector3(...connection.from),
                new THREE.Vector3(...connection.to)
            ];
            
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const material = new THREE.LineBasicMaterial({ 
                color: 0x32b8c6,
                linewidth: 3,
                transparent: true,
                opacity: 0.7
            });
            
            const line = new THREE.Line(geometry, material);
            line.userData = {
                name: `Data Flow Connection ${index + 1}`,
                type: 'connection'
            };
            
            this.components.set(`connection-${index}`, line);
            this.scene.add(line);
        });
    }
    
    setupEventListeners() {
        // Panel toggles
        document.getElementById('toggle-panel').addEventListener('click', () => {
            document.getElementById('control-panel').classList.toggle('collapsed');
        });
        
        document.getElementById('toggle-info').addEventListener('click', () => {
            document.getElementById('info-panel').classList.toggle('collapsed');
        });
        
        // Component visibility toggles
        document.getElementById('toggle-embeddings').addEventListener('change', (e) => {
            this.toggleComponentVisibility('embedding', e.target.checked);
        });
        
        document.getElementById('toggle-attention').addEventListener('change', (e) => {
            this.toggleComponentVisibility('attention', e.target.checked);
        });
        
        document.getElementById('toggle-mlp').addEventListener('change', (e) => {
            this.toggleComponentVisibility('mlp', e.target.checked);
        });
        
        document.getElementById('toggle-layernorm').addEventListener('change', (e) => {
            this.toggleComponentVisibility('normalization', e.target.checked);
        });
        
        document.getElementById('toggle-output').addEventListener('change', (e) => {
            this.toggleComponentVisibility('output', e.target.checked);
        });
        
        document.getElementById('toggle-connections').addEventListener('change', (e) => {
            this.toggleComponentVisibility('connection', e.target.checked);
        });
        
        // Animation controls
        document.getElementById('play-animation').addEventListener('click', () => {
            this.startAnimation();
        });
        
        document.getElementById('pause-animation').addEventListener('click', () => {
            this.pauseAnimation();
        });
        
        document.getElementById('reset-animation').addEventListener('click', () => {
            this.resetAnimation();
        });
        
        document.getElementById('animation-speed').addEventListener('input', (e) => {
            this.animationSpeed = parseFloat(e.target.value);
        });
        
        // View presets
        document.getElementById('view-overview').addEventListener('click', () => {
            this.setViewPreset('overview');
        });
        
        document.getElementById('view-attention').addEventListener('click', () => {
            this.setViewPreset('attention');
        });
        
        document.getElementById('view-layers').addEventListener('click', () => {
            this.setViewPreset('layers');
        });
        
        document.getElementById('view-flow').addEventListener('click', () => {
            this.setViewPreset('flow');
        });
        
        // Educational mode
        document.getElementById('education-mode').addEventListener('click', () => {
            this.toggleEducationalMode();
        });
        
        // Resize handler
        window.addEventListener('resize', () => {
            this.onWindowResize();
        });
        
        // Mouse hover for component info
        this.renderer.domElement.addEventListener('mousemove', (event) => {
            this.onMouseMove(event);
        });
    }
    
    toggleComponentVisibility(type, visible) {
        this.components.forEach((component, key) => {
            if (component.userData && component.userData.type === type) {
                component.visible = visible;
            }
        });
    }
    
    startAnimation() {
        this.isAnimating = true;
        this.animateDataFlow();
    }
    
    pauseAnimation() {
        this.isAnimating = false;
    }
    
    resetAnimation() {
        this.isAnimating = false;
        // Reset all component positions and states
        this.components.forEach(component => {
            if (component.userData && component.userData.originalPosition) {
                component.position.copy(component.userData.originalPosition);
            }
        });
    }
    
    animateDataFlow() {
        if (!this.isAnimating) return;
        
        // Simple animation example - pulse effect on active components
        const time = Date.now() * 0.001 * this.animationSpeed;
        
        this.components.forEach(component => {
            if (component.userData && component.userData.type === 'attention') {
                component.rotation.y = time * 0.5;
            }
            if (component.userData && component.userData.type === 'mlp') {
                component.scale.y = 1 + Math.sin(time * 2) * 0.1;
            }
        });
        
        requestAnimationFrame(() => this.animateDataFlow());
    }
    
    setViewPreset(preset) {
        switch (preset) {
            case 'overview':
                this.camera.position.set(0, 15, 25);
                this.camera.lookAt(0, 0, 0);
                break;
            case 'attention':
                this.camera.position.set(10, 0, 10);
                this.camera.lookAt(0, 0, 0);
                break;
            case 'layers':
                this.camera.position.set(0, 0, 30);
                this.camera.lookAt(0, 0, 0);
                break;
            case 'flow':
                this.camera.position.set(15, 5, 15);
                this.camera.lookAt(0, 0, 0);
                break;
        }
    }
    
    toggleEducationalMode() {
        this.educationalMode = !this.educationalMode;
        const button = document.getElementById('education-mode');
        button.textContent = this.educationalMode ? '🎓 Expert Mode' : '📚 Educational Mode';
        
        // Show/hide additional educational elements
        if (this.educationalMode) {
            this.addEducationalElements();
        } else {
            this.removeEducationalElements();
        }
    }
    
    addEducationalElements() {
        // Add labels and explanations to components
        this.components.forEach((component, key) => {
            if (component.userData && component.userData.name) {
                // Add 3D text labels (simplified version)
                const label = this.createTextLabel(component.userData.name);
                label.position.copy(component.position);
                label.position.y += 2;
                component.add(label);
            }
        });
    }
    
    removeEducationalElements() {
        // Remove educational elements
        this.components.forEach(component => {
            const labels = component.children.filter(child => child.userData.isLabel);
            labels.forEach(label => component.remove(label));
        });
    }
    
    createTextLabel(text) {
        // Simple text label using sprites
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 256;
        canvas.height = 64;
        
        context.fillStyle = 'rgba(255, 255, 255, 0.9)';
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.fillStyle = 'black';
        context.font = '12px Arial';
        context.textAlign = 'center';
        context.fillText(text, canvas.width / 2, canvas.height / 2);
        
        const texture = new THREE.CanvasTexture(canvas);
        const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.scale.set(4, 1, 1);
        sprite.userData.isLabel = true;
        
        return sprite;
    }
    
    onMouseMove(event) {
        // Simple hover detection for component info
        const rect = this.renderer.domElement.getBoundingClientRect();
        const mouse = new THREE.Vector2(
            ((event.clientX - rect.left) / rect.width) * 2 - 1,
            -((event.clientY - rect.top) / rect.height) * 2 + 1
        );
        
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, this.camera);
        
        const intersects = raycaster.intersectObjects(this.scene.children, true);
        
        if (intersects.length > 0) {
            const intersectedObject = intersects[0].object;
            let component = intersectedObject;
            
            // Find parent component with userData
            while (component && !component.userData.name) {
                component = component.parent;
            }
            
            if (component && component.userData.name) {
                this.updateComponentInfo(component.userData);
            }
        }
    }
    
    updateComponentInfo(userData) {
        document.getElementById('component-name').textContent = userData.name || 'Unknown Component';
        document.getElementById('component-description').textContent = userData.description || 'No description available';
        
        const specs = document.getElementById('component-specs');
        specs.innerHTML = '';
        
        if (userData.parameters) {
            specs.innerHTML += `
                <div class="spec-item">
                    <span class="spec-label">Parameters:</span>
                    <span class="spec-value">${userData.parameters}</span>
                </div>
            `;
        }
        
        if (userData.layer !== undefined) {
            specs.innerHTML += `
                <div class="spec-item">
                    <span class="spec-label">Layer:</span>
                    <span class="spec-value">${userData.layer + 1}</span>
                </div>
            `;
        }
        
        if (userData.type) {
            specs.innerHTML += `
                <div class="spec-item">
                    <span class="spec-label">Type:</span>
                    <span class="spec-value">${userData.type}</span>
                </div>
            `;
        }
    }
    
    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Update camera controls
        this.controls.mouseX += (this.controls.targetX - this.controls.mouseX) * 0.05;
        this.controls.mouseY += (this.controls.targetY - this.controls.mouseY) * 0.05;
        this.controls.distance += (this.controls.targetDistance - this.controls.distance) * 0.05;
        
        // Update camera position
        this.camera.position.x = Math.sin(this.controls.mouseX) * this.controls.distance;
        this.camera.position.z = Math.cos(this.controls.mouseX) * this.controls.distance;
        this.camera.position.y = Math.sin(this.controls.mouseY) * this.controls.distance * 0.5 + 5;
        this.camera.lookAt(0, 0, 0);
        
        this.renderer.render(this.scene, this.camera);
    }
}

// Initialize the visualizer when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const visualizer = new NanoGPTVisualizer();
});