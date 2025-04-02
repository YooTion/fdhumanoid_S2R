import pymeshlab


def calculate_inertial_tag(file_name=None, mass=-1, pr=8, scale_factor=100):
    ms = pymeshlab.MeshSet()

    if file_name is None:
        print('Please put the input file to the same folder as this script and type in the full name of your file.')
        file_name = input()
    ms.load_new_mesh(file_name)

    if mass < 0:
        print('Please type the mass of your object in kg')
        mass = float(input())

    print('Calculating the center of mass')
    geom = ms.get_geometric_measures()
    com = geom['barycenter']

    print('Scaling the mesh')
    ms.compute_matrix_from_scaling_or_normalization(axisx=scale_factor, axisy=scale_factor, axisz=scale_factor)

    print('Generating the convex hull of the mesh')
    ms.generate_convex_hull()  # TODO only if object is not watertight

    print('Calculating intertia tensor')
    geom = ms.get_geometric_measures()
    volume = geom['mesh_volume']
    tensor = geom['inertia_tensor'] / pow(scale_factor, 2) * mass / volume

    intertial_xml = f'<inertial>\n  <origin xyz="{com[0]:.{pr}f} {com[1]:.{pr}f} {com[2]:.{pr}f}"/>\n  <mass value="{mass:.{pr}f}"/>\n  <inertia ixx="{tensor[0, 0]:.{pr}f}" ixy="{tensor[1, 0]:.{pr}f}" ixz="{tensor[2, 0]:.{pr}f}" iyy="{tensor[1, 1]:.{pr}f}" iyz="{tensor[1, 2]:.{pr}f}" izz="{tensor[2, 2]:.{pr}f}"/>\n</inertial>'
    print(intertial_xml)


if __name__ == '__main__':
    calculate_inertial_tag("/home/ubuntu/current_work/huma_robot_gym/resources/robots/T1Bot/meshes/R_ankleP_Link.STL", mass=0.32 ,pr=8, scale_factor=100)  # TODO command line arguments
    calculate_inertial_tag("/home/ubuntu/current_work/huma_robot_gym/resources/robots/T1Bot/meshes/L_ankleP_Link.STL", mass=0.32 ,pr=8, scale_factor=100)  # TODO command line arguments
    calculate_inertial_tag("/home/ubuntu/current_work/huma_robot_gym/resources/robots/T1Bot/meshes/torsoP_Link.STL", mass=9.46 ,pr=8, scale_factor=100)  # TODO command line arguments
    calculate_inertial_tag("/home/ubuntu/current_work/huma_robot_gym/resources/robots/T1Bot/meshes/base_link.STL", mass=2.73 ,pr=8, scale_factor=100)  # TODO command line arguments
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/base_link.STL", mass=2.73 ,pr=8, scale_factor=100)  # TODO command line arguments
    # print("=================================================")
    
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/L_hipP_Link.STL", mass=1.54 ,pr=8, scale_factor=100)  # TODO command line arguments
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/R_hipP_Link.STL", mass=1.54 ,pr=8, scale_factor=100)  # TODO command line arguments
    # print("=================================================")
    
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/L_hipR_Link.STL", mass=2.13 ,pr=8, scale_factor=100)  # TODO command line arguments
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/R_hipR_Link.STL", mass=2.13 ,pr=8, scale_factor=100)  # TODO command line arguments
    # print("=================================================")
    
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/L_hipY_Link.STL", mass=0.135 ,pr=8, scale_factor=100)  # TODO command line arguments
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/R_hipY_Link.STL", mass=0.135 ,pr=8, scale_factor=100)  # TODO command line arguments
    # print("=================================================")
    
    
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/L_knee_Link.STL", mass=3.02 ,pr=8, scale_factor=100)  # TODO command line arguments
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/R_knee_Link.STL", mass=3.02 ,pr=8, scale_factor=100)  # TODO command line arguments
    # print("=================================================")
    
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/L_ankleY_Link.STL", mass=0.11 ,pr=8, scale_factor=100)  # TODO command line arguments
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/R_ankleY_Link.STL", mass=0.11 ,pr=8, scale_factor=100)  # TODO command line arguments
    # print("=================================================")
    
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/L_ankleR_Link.STL", mass=0.032 ,pr=8, scale_factor=100)  # TODO command line arguments
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/R_ankleR_Link.STL", mass=0.032 ,pr=8, scale_factor=100)  # TODO command line arguments
    # print("=================================================")
    
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/L_ankleP_Link.STL", mass=0.38 ,pr=8, scale_factor=100)  # TODO command line arguments
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/R_ankleP_Link.STL", mass=0.38 ,pr=8, scale_factor=100)  # TODO command line arguments
    # print("=================================================")
    
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/L_sole_Link.STL", mass=0.089 ,pr=8, scale_factor=100)  # TODO command line arguments
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/R_sole_Link.STL", mass=0.089 ,pr=8, scale_factor=100)  # TODO command line arguments
    # print("=================================================")
    
    
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/torsoY_Link.STL", mass=1.38 ,pr=8, scale_factor=100)  # TODO command line arguments
    # print("=================================================")
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/torsoR_Link.STL", mass=2.13 ,pr=8, scale_factor=100)  # TODO command line arguments
    # print("=================================================")
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/torsoP_Link.STL", mass=9.46 ,pr=8, scale_factor=100)  # TODO command line arguments
    # print("=================================================")
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/L_shoulderY_Link.STL", mass=0.064 ,pr=8, scale_factor=100)  # TODO command line arguments
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/R_shoulderY_Link.STL", mass=0.064 ,pr=8, scale_factor=100)  # TODO command line arguments
    # print("=================================================")
    
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/L_shoulderR_Link.STL", mass=0.77,pr=8, scale_factor=100)  # TODO command line arguments
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/R_shoulderR_Link.STL", mass=0.77 ,pr=8, scale_factor=100)  # TODO command line arguments
    # print("=================================================")
    
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/L_elbowY_Link.STL", mass=0.673,pr=8, scale_factor=100)  # TODO command line arguments
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/R_elbowY_Link.STL", mass=0.673 ,pr=8, scale_factor=100)  # TODO command line arguments
    # print("=================================================")
    
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/L_elbowR_Link.STL", mass=1.373 ,pr=8, scale_factor=100)  # TODO command line arguments
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/R_elbowR_Link.STL", mass=1.373 ,pr=8, scale_factor=100)  # TODO command line arguments
    # print("=================================================")
    
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/neckR_Link.STL", mass=0.0068, pr=8, scale_factor=100)  # TODO command line arguments
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/neckP_Link.STL", mass=0.53 ,pr=8, scale_factor=100)  # TODO command line arguments
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/headR_Link.STL", mass=0.439,pr=8, scale_factor=100)  # TODO command line arguments
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/headP_Link.STL", mass=0.102 ,pr=8, scale_factor=100)  # TODO command line arguments
    # calculate_inertial_tag("/home/ubuntu/current_work/fd_model_op/meshes/headY_Link.STL", mass=0.418 ,pr=8, scale_factor=100)  # TODO command line arguments
    # print("=================================================")
    