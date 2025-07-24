def run_render_pipeline(pkl_path):
    import os
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    import numpy as np
    import pickle
    import torch
    import pyrender
    import trimesh
    import cv2
    import smplx

    # === Load SMPL Model ===
    model = smplx.SMPL(model_path="SMPL_NEUTRAL.pkl", gender='neutral', batch_size=1)
    
    # === Load .pkl motion file ===
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    poses = data["smpl_poses"]  # (T, 72)
    trans = data.get("smpl_trans", np.zeros((poses.shape[0], 3)))  # fallback
    T = poses.shape[0]
    
    # === Set up renderer ===
    scene = pyrender.Scene()
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0,  0.0,  0.0,  0.0],     # Right
        [0.0,  0.0, -1.0, -3.5],     # Down
        [0.0,  1.0,  0.0,  2.5],     # Backward
        [0.0,  0.0,  0.0,  1.0]
    ])
    scene.add(camera, pose=camera_pose)
    
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
    scene.add(light)

    renderer = pyrender.OffscreenRenderer(512, 512)

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    output_video_path = os.path.join("outputs", os.path.basename(pkl_path).replace(".pkl", ".mp4"))
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (512, 512))
    
    # === Render each frame ===
    for i in range(T):
        pose = torch.tensor(poses[i:i+1], dtype=torch.float32)
        transl = torch.tensor(trans[i:i+1], dtype=torch.float32)
    
        output = model(
            global_orient=pose[:, :3],
            body_pose=pose[:, 3:],
            transl=transl,
            betas=torch.zeros(1, 10),
            return_verts=True
        )
    
        verts = output.vertices[0].detach().cpu().numpy()
        mesh = trimesh.Trimesh(vertices=verts, faces=model.faces, process=False)
        render_mesh = pyrender.Mesh.from_trimesh(mesh)
    
        # Remove previous mesh and add new one
        for node in list(scene.get_nodes()):
            if isinstance(node.mesh, pyrender.Mesh):
                scene.remove_node(node)
        scene.add(render_mesh)
    
        color, _ = renderer.render(scene)
        video.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    
    video.release()
    renderer.delete()
    print("Rendering complete:", output_video_path)

    return f"/{output_video_path}"
