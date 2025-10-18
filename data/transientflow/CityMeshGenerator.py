import gmsh
import math
import numpy as np
import random
import meshio
import argparse

def compute_centroid(points):
    centroid = [sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points)]
    return centroid

def polar_angle(point,centroid):
    x, y = point
    angle = math.atan2(y - centroid[1], x - centroid[0])
    return angle

def pair_neighboring_elements(numbers):
    paired_list = []
    length = len(numbers)

    for i in range(length):
        pair = [numbers[i], numbers[(i + 1) % length]]
        paired_list.append(pair)

    return paired_list

def sort_points(points):
    centroid = compute_centroid(points)
    p_keys = [-polar_angle(p,centroid) for p in points]
    sorted_indices = list(np.argsort(p_keys))
    points = [points[s] for s in sorted_indices]

    return points

def compute_distance(p1,p2):
    d = (p1[0]-p2[0])**2+(p1[1]-p2[1])**2
    return d

def triangle_area(x1, y1, x2, y2, x3, y3):
    # Shoelace Formula
    area = 0.5 * np.abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    return area

def generate_mesh(output_filename, n_objects):
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add('model')
    
    # 计算域尺寸
    domain_xmin, domain_xmax = -0.5, 1.0
    domain_ymin, domain_ymax = -0.5, 0.5
    domain_zmin, domain_zmax = 0.0, 0.5
    
    # 直接创建三维计算域长方体
    domain = gmsh.model.occ.addBox(
        domain_xmin, domain_ymin, domain_zmin,
        domain_xmax - domain_xmin,
        domain_ymax - domain_ymin,
        domain_zmax - domain_zmin
    )
    
    # 创建长方体障碍物（高楼）
    buildings = []  # 保存建筑的位置和尺寸信息
    building_top_faces = []  # 存储所有建筑顶面
    all_building_faces = []  # 存储所有建筑表面
    
    # 流体域初始为整个计算域
    fluid_domain = [(3, domain)]
    
    for i in range(n_objects):
        # 随机生成长方体参数
        width = random.uniform(0.05, 0.15)
        height = random.uniform(0.05, domain_zmax - 0.01)  # 高度不超过计算域高度
        depth = random.uniform(0.05, 0.15)
        center_x = random.uniform(domain_xmin + width/2 + 0.01, domain_xmax - width/2 - 0.01)
        center_y = random.uniform(domain_ymin + depth/2 + 0.01, domain_ymax - depth/2 - 0.01)
        
        # 直接创建三维长方体
        building = gmsh.model.occ.addBox(
            center_x - width/2, 
            center_y - depth/2, 
            domain_zmin, 
            width, 
            depth, 
            height
        )
        
        # 保存建筑信息（位置和尺寸）
        buildings.append({
            'center_x': center_x,
            'center_y': center_y,
            'width': width,
            'depth': depth,
            'height': height,
            'zmin': domain_zmin,
            'zmax': domain_zmin + height
        })
        
        # 从流体域中减去长方体
        fluid_domain, _ = gmsh.model.occ.cut(
            fluid_domain, 
            [(3, building)], 
            removeObject=True, 
            removeTool=True
        )
        
        print(f"Added building {i+1}/{n_objects}: size={width}x{depth}x{height}")
    
    # 同步几何模型
    gmsh.model.occ.synchronize()
    
    # 获取流体域的所有边界（包括内部边界）
    fluid_boundaries = gmsh.model.getBoundary(fluid_domain, oriented=False)
    
    # 根据建筑信息识别建筑表面
    building_faces = []
    for face in fluid_boundaries:
        # 获取面的中心点
        tag = face[1]
        center = gmsh.model.occ.getCenterOfMass(face[0], tag)
        x, y, z = center
        
        # 检查这个中心点是否在任何一个建筑的包围盒内
        for building in buildings:
            cx = building['center_x']
            cy = building['center_y']
            w = building['width']
            d = building['depth']
            zmin = building['zmin']
            zmax = building['zmax']
            
            if (cx - w/2 - 1e-5 <= x <= cx + w/2 + 1e-5 and
                cy - d/2 - 1e-5 <= y <= cy + d/2 + 1e-5 and
                zmin - 1e-5 <= z <= zmax + 1e-5):
                # 这个面属于当前建筑
                building_faces.append(tag)
                break
    
    # 获取计算域的边界
    domain_boundaries = gmsh.model.getBoundary([(3, domain)])
    domain_faces = [b[1] for b in domain_boundaries]
    
    # 根据位置识别边界类型
    front_faces = []
    back_faces = []
    outflow_faces = []
    inflow_faces = []
    sidewall_faces = []
    
    for face in domain_faces:
        face = abs(face)
        if face in building_faces:
            continue  # 跳过建筑表面
        center = gmsh.model.occ.getCenterOfMass(2, face)
        x, y, z = center
        
        # 前平面 (z = min)
        if abs(z - domain_zmin) < 1e-5:
            front_faces.append(face)  
        # 后平面 (z = max)
        elif abs(z - domain_zmax) < 1e-5:
            back_faces.append(face)
        # 出口 (x = max)
        elif abs(x - domain_xmax) < 1e-5:
            outflow_faces.append(face)
        # 入口 (x = min)
        elif abs(x - domain_xmin) < 1e-5:
            inflow_faces.append(face)
        # 侧壁 (y = min 或 max)
        elif abs(y - domain_ymin) < 1e-5 or abs(y - domain_ymax) < 1e-5:
            sidewall_faces.append(face)
    
    if not (len(front_faces) == 1 and len(back_faces) == 1 and len(outflow_faces) == 1 and len(inflow_faces) == 1 and len(sidewall_faces) == 2):
        print("Error: Boundary face identification may be incorrect.")
        print(f"Front faces: {front_faces}")
        print(f"Back faces: {back_faces}")
        print(f"Outflow faces: {outflow_faces}")
        print(f"Inflow faces: {inflow_faces}")
        print(f"Sidewall faces: {sidewall_faces}")
        raise ValueError("Boundary face identification failed.")
    
    # 物理组命名
    gmsh.model.addPhysicalGroup(2, front_faces, name="FrontPlane")
    gmsh.model.addPhysicalGroup(2, back_faces, name="BackPlane")
    gmsh.model.addPhysicalGroup(2, outflow_faces, name="outflow")
    gmsh.model.addPhysicalGroup(2, inflow_faces, name="inflow")
    gmsh.model.addPhysicalGroup(2, sidewall_faces, name="sidewalls")
    
    # 建筑墙面（包括顶面）
    gmsh.model.addPhysicalGroup(2, building_faces, name="wall")
    
    # 流体域
    fluid_volumes = [v[1] for v in fluid_domain]
    gmsh.model.addPhysicalGroup(3, fluid_volumes, name="internal")
    
    # 网格设置
    res_min = random.uniform(0.01, 0.02)
    
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "SurfacesList", building_faces)
    gmsh.model.mesh.field.setNumber(1, "Sampling", 100)
    
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", res_min)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", res_min * 2.9)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 0.2)
    
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    
    # 设置网格算法（推荐使用Netgen）
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # 6 = Netgen
    
    gmsh.model.mesh.generate(3)
    
    gmsh.write(output_filename)
    
    # 网格质量检查
    msh = meshio.read(output_filename)
    print(f"Mesh generated with {len(msh.points)} points")
    
    # 计算三角形面积（仅用于质量检查）
    if 'triangle' in msh.cells_dict:
        triangles = msh.cells_dict['triangle']
        if len(triangles) > 0:
            t = msh.points[triangles]
            x1 = t[:, 0, 0]
            y1 = t[:, 0, 1]
            x2 = t[:, 1, 0]
            y2 = t[:, 1, 1]
            x3 = t[:, 2, 0]
            y3 = t[:, 2, 1]
            area = triangle_area(x1, y1, x2, y2, x3, y3)
            print(f"Mesh area ratio: {area.max()/area.min() if area.min() > 0 else 'N/A'}")
    
    gmsh.finalize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_filename', type=str, help='output filename of mesh')
    parser.add_argument('n_objects', type=int, help='number of buildings in case')
    args = parser.parse_args()
    
    generate_mesh(args.output_filename, args.n_objects)