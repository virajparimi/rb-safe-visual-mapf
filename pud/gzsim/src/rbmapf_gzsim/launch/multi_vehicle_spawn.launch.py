import os
import shutil
import yaml
import numpy as np
from pathlib import Path
from copy import deepcopy
from jinja2 import Environment, FileSystemLoader

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from ament_index_python import get_package_share_directory
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import (
    TimerAction,
    OpaqueFunction,
    ExecuteProcess,
    RegisterEventHandler,
    DeclareLaunchArgument,
    IncludeLaunchDescription
)

PACKAGE_NAME = 'rbmapf_gzsim'

# Source the ros opt
# Source the px4_msgs/ ros2_ws / crazyswarm_ws
# Colcon build this code
# Source this code
# Source the gazebo-classic inside PX4 -->
#   source ~/Developer/PX4-Autopilot/Tools/simulation/gazebo-classic/setup_gazebo.bash
#   ~/Developer/PX4-Autopilot/ ~/Developer/PX4-Autopilot/build/px4_sitl_default/
# Run the ros2 launch script


def resolve_package_asset(path_str):
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return Path(get_package_share_directory(PACKAGE_NAME)) / path


def generated_asset_paths(source_world_file, generated_assets_dir):
    source_world_file = Path(source_world_file)
    output_dir = Path(generated_assets_dir).expanduser()
    base_name = source_world_file.stem
    suffix = source_world_file.suffix.lower()
    return {
        'output_dir': output_dir,
        'world': output_dir / f'{base_name}_walls{suffix}',
        'starts': output_dir / f'{base_name}_starts.txt',
        'bounds': output_dir / f'{base_name}_bounds.txt',
        'walls_matrix': output_dir / f'{base_name}_walls_matrix.npy',
    }


def prepare_generated_model_dir(template_path, generated_assets_dir):
    template_dir = Path(template_path).parent
    rendered_dir = Path(generated_assets_dir).expanduser() / template_dir.name
    rendered_dir.mkdir(parents=True, exist_ok=True)

    meshes_src = template_dir / 'meshes'
    meshes_dst = rendered_dir / 'meshes'
    if meshes_src.is_dir():
        shutil.copytree(meshes_src, meshes_dst, dirs_exist_ok=True)

    model_config_src = template_dir / 'model.config'
    model_config_dst = rendered_dir / 'model.config'
    if model_config_src.is_file():
        shutil.copy2(model_config_src, model_config_dst)

    return rendered_dir


def append_env_path(env, key, path):
    path_str = str(Path(path).expanduser())
    current = env.get(key)
    if current:
        env[key] = f"{path_str}{os.pathsep}{current}"
    else:
        env[key] = path_str


def generate_rviz_config(base_config_path, output_path, namespaces):
    config = yaml.safe_load(Path(base_config_path).read_text())
    displays = config["Visualization Manager"]["Displays"]

    def topic_value(display):
        topic = display.get("Topic")
        return topic.get("Value") if isinstance(topic, dict) else None

    path_template = next(
        display for display in displays
        if display.get("Class") == "rviz_default_plugins/Path"
    )
    marker_template = next(
        display for display in displays
        if display.get("Class") == "rviz_default_plugins/MarkerArray"
        and topic_value(display) == "/crazyflie_1/waypoint_markers"
    )
    static_displays = [
        display
        for display in displays
        if not (
            (topic_value(display) or "").endswith("/trajectory")
            or (topic_value(display) or "").endswith("/waypoint_markers")
        )
    ]

    path_colors = [
        "255; 0; 0",
        "0; 255; 0",
        "0; 0; 255",
        "255; 255; 0",
        "255; 0; 255",
        "0; 255; 255",
        "255; 128; 0",
        "128; 0; 255",
    ]

    drone_displays = []
    for idx, namespace in enumerate(namespaces):
        path_display = deepcopy(path_template)
        path_display["Name"] = f"Path {idx + 1}"
        path_display["Color"] = path_colors[idx % len(path_colors)]
        path_display["Topic"]["Value"] = f"{namespace}/trajectory"
        drone_displays.append(path_display)

        marker_display = deepcopy(marker_template)
        marker_display["Name"] = f"Waypoints {idx + 1}"
        marker_display["Topic"]["Value"] = f"{namespace}/waypoint_markers"
        drone_displays.append(marker_display)

    merged_displays = []
    inserted_drone_displays = False
    for display in static_displays:
        merged_displays.append(display)
        if not inserted_drone_displays and display.get("Class") == "rviz_default_plugins/Grid":
            merged_displays.extend(drone_displays)
            inserted_drone_displays = True
    if not inserted_drone_displays:
        merged_displays = drone_displays + merged_displays

    config["Visualization Manager"]["Displays"] = merged_displays
    output_path = Path(output_path)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return str(output_path)


def spawn_drones(context, agent_idx, agent_x, agent_y, agent_z=1.0):
    actions = []

    px4_src = LaunchConfiguration('px4_src_path').perform(context)
    px4_target = LaunchConfiguration('px4_target').perform(context)

    px4_build_path = f"{px4_src}/build/{px4_target}"

    px4_bin = f"{px4_build_path}/bin/px4"
    px4_work_dir = f"{px4_build_path}/rootfs/{agent_idx}"

    px4 = ExecuteProcess(
        cmd=[px4_bin, '-i', str(agent_idx), '-d',  f'{px4_build_path}/etc'],
        cwd=px4_work_dir,
        name=f'px4_{agent_idx}_sitl',
        output='screen'
    )
    actions.append(px4)

    jinja_cmd = ExecuteProcess(
        cmd=[
            'python3',
            px4_src + '/Tools/simulation/gazebo-classic/sitl_gazebo-classic/scripts/jinja_gen.py',
            px4_src + '/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/iris/iris.sdf.jinja',
            px4_src + '/Tools/simulation/gazebo-classic/sitl_gazebo-classic',
            '--mavlink_tcp_port', str(4560 + agent_idx),
            '--mavlink_udp_port', str(14560 + agent_idx),
            '--mavlink_id', str(1 + agent_idx),
            '--gst_udp_port', str(5600 + agent_idx),
            '--video_uri', str(5600 + agent_idx),
            '--mavlink_cam_udp_port', str(14530 + agent_idx),
            '--output-file', f'/tmp/iris_{agent_idx}.sdf',
        ],
        name=f'jinja_{agent_idx}_cmd',
        output='screen',
    )
    actions.append(jinja_cmd)

    gz_cmd = ExecuteProcess(
        cmd=['gz', 'model',
             '--spawn-file', f'/tmp/iris_{agent_idx}.sdf',
             '--model-name', f'iris_{agent_idx}',
             '-x', str(agent_x),
             '-y', str(agent_y),
             '-z', str(agent_z)
             ],
        name='gz_spawn',
        output='screen',
    )
    actions.append(gz_cmd)
    return actions


def render_crazyflie_models(context, *args, **kwargs):
    num_drones = int(LaunchConfiguration('num_drones').perform(context))
    template_path = resolve_package_asset(LaunchConfiguration('model_jinja').perform(context))
    generated_assets_dir = Path(LaunchConfiguration('generated_assets_dir').perform(context)).expanduser()
    generated_assets_dir.mkdir(parents=True, exist_ok=True)
    rendered_model_dir = prepare_generated_model_dir(template_path, generated_assets_dir)

    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        keep_trailing_newline=True,
        autoescape=False,
    )
    template = env.get_template(template_path.name)

    out_paths = []
    for drone_id in range(1, num_drones + 1):
        ns = f'crazyflie_{drone_id}'
        rendered = template.render(namespace=ns)
        out_file = rendered_model_dir / f'{ns}.sdf'
        out_file.write_text(rendered)
        out_paths.append(str(out_file))

    context.launch_configurations['rendered_sdfs'] = ';'.join(out_paths)
    return []


def spawn_crazyflies(context, starts):
    num_drones = int(LaunchConfiguration('num_drones').perform(context))
    sdfs = context.launch_configurations['rendered_sdfs'].split(';')
    z_start = 0.1
    actions = []
    for idx in range(1, num_drones + 1):
        x, y = float(starts[idx - 1, 0]), float(starts[idx - 1, 1])
        ns = Path(sdfs[idx - 1]).stem
        spawn = Node(
            package='ros_gz_sim',
            executable='create',
            name=f'spawn_{ns}',
            arguments=[
                '--world', 'default_walls',
                '--file',   f'file://{sdfs[idx - 1]}',
                '--name',   ns,
                '--allow-renaming', 'true',
                '--x', str(x), '--y', str(y), '--z', str(z_start),
            ],
            output='screen'
        )
        actions.append(spawn)
    return actions


def spawn_processes(context, *args, **kwargs):
    gz_version = LaunchConfiguration('gz_version').perform(context)
    world_file = resolve_package_asset(LaunchConfiguration('world_file').perform(context))
    generated_assets_dir = Path(LaunchConfiguration('generated_assets_dir').perform(context)).expanduser()
    generated_assets_dir.mkdir(parents=True, exist_ok=True)
    num_drones = int(LaunchConfiguration('num_drones').perform(context))
    config_file = LaunchConfiguration('config_file').perform(context)
    habitat = LaunchConfiguration('habitat').perform(context) == 'True'
    use_crazyflies = LaunchConfiguration('use_crazyflies').perform(context)
    problem_set_file = LaunchConfiguration('problem_set_file').perform(context)
    hardware_demo = LaunchConfiguration('use_hardware').perform(context) == 'True'
    constrained_ckpt_file = LaunchConfiguration('constrained_ckpt_file').perform(context)
    unconstrained_ckpt_file = LaunchConfiguration('unconstrained_ckpt_file').perform(context)

    generated_paths = generated_asset_paths(world_file, generated_assets_dir)
    starts_file = generated_paths['starts']
    starts = np.loadtxt(starts_file, delimiter=',')
    if starts.ndim == 1:
        starts = starts.reshape(1, -1)
    if starts.shape[0] != num_drones:
        raise RuntimeError("Mismatch in start count")
    walls_file = generated_paths['walls_matrix']

    if habitat:
        bounds_file = generated_paths['bounds']

    z_start = 1.0
    prev_action = None
    use_sim_time = not hardware_demo
    actions = []

    if use_crazyflies == 'False':
        if gz_version == 'classic':
            updated_world_file = generated_paths['world']
            gzserver_cmd = ExecuteProcess(
                cmd=[
                    'gzserver',
                    str(updated_world_file),
                    '--verbose',
                    '-s',
                    'libgazebo_ros_init.so',
                    '-s',
                    'libgazebo_ros_factory.so'
                ],
                name='gzserver_launch',
                output='screen'
            )
            actions.append(gzserver_cmd)

            for idx in range(1, num_drones + 1):
                x, y = float(starts[idx-1, 0]), float(starts[idx-1, 1])
                spawn_drone_cmd = OpaqueFunction(function=spawn_drones, args=[idx, x, y, z_start])
                timed_action = TimerAction(period=5.0, actions=[spawn_drone_cmd])
                actions.append(timed_action)

            gzclient_cmd = ExecuteProcess(
                cmd=['gzclient'],
                name='gzclient_cmd',
                output='screen',
            )
            actions.append(RegisterEventHandler(
                OnProcessStart(target_action=gzserver_cmd, on_start=[gzclient_cmd]),
            ))
            prev_action = gzclient_cmd

        elif gz_version == 'harmonic' and use_crazyflies == 'False':
            env = os.environ.copy()
            env['PX4_SIM_MODEL'] = 'gz_x500'
            env['PX4_GZ_WORLD'] = generated_paths['world'].stem
            px4_bin = LaunchConfiguration('px4_bin_path').perform(context)

            for idx in range(num_drones):
                x, y = float(starts[idx, 0]), float(starts[idx, 1])
                env['PX4_GZ_MODEL_POSE'] = f"{x},{y},{z_start}"

                px4 = ExecuteProcess(
                    cmd=[px4_bin, '-i', str(idx+1)],
                    env=env,  # type: ignore
                    name=f'px4_{idx+1}_sitl',
                    output='screen'
                )

                if idx == 0:
                    actions.append(px4)
                    prev_action = px4
                else:
                    timed_action = TimerAction(
                        period=3.0,
                        actions=[px4],
                    )
                    actions.append(
                        timed_action
                    )
                    prev_action = px4

        if num_drones > 0:
            microxrce_agent_cmd = ExecuteProcess(
                cmd=[
                    'MicroXRCEAgent', 'udp4', '--port', '8888'
                ],
                name='microxrce_agent',
                output='screen'
            )
            actions.append(RegisterEventHandler(
                OnProcessStart(target_action=prev_action, on_start=[microxrce_agent_cmd]),
            ))

            waypoint_generator = Node(
                package="rbmapf_gzsim",
                executable="waypoint_generator",
                name="waypoint_generator",
                output="screen",
                arguments=[
                    '--visual', str(habitat),
                    '--config_file', config_file,
                    '--num_agents', str(num_drones),
                    '--problem_set_file', problem_set_file,
                    '--constrained_ckpt_file', constrained_ckpt_file,
                    '--unconstrained_ckpt_file', unconstrained_ckpt_file,
                ],
                parameters=[{
                    'risk_bound_percent': ParameterValue(
                        LaunchConfiguration('risk_bound_percent'),
                        value_type=float,
                    ),
                    'problem_index': ParameterValue(
                        LaunchConfiguration('problem_index'),
                        value_type=int,
                    ),
                }],
            )
            actions.append(RegisterEventHandler(
                OnProcessStart(target_action=microxrce_agent_cmd, on_start=[waypoint_generator]),
            ))

            drone_ids = ','.join(str(i + 2) for i in range(num_drones))
            drone_namespaces = [f'/px4_{i + 1}' for i in range(num_drones)]
            drone_namespaces_csv = ','.join(drone_namespaces)
            px4_drone_control = Node(
                package="rbmapf_gzsim",
                executable="px4_drone_control",
                name="px4_drone_control",
                output="screen",
                arguments=[
                    '--drone_ids', drone_ids,
                    '--drone_namespaces', drone_namespaces_csv,
                    '--config_file', config_file,
                    '--ckpt_file', constrained_ckpt_file,
                    '--walls_file', walls_file,
                    '--gz_version', gz_version,
                ]
            )
            actions.append(RegisterEventHandler(
                OnProcessStart(target_action=microxrce_agent_cmd, on_start=[px4_drone_control]),
            ))

            rviz_config = os.path.join(
                get_package_share_directory('rbmapf_gzsim'),
                'launch',
                'multi_drone.rviz'
            )
            rviz_config = generate_rviz_config(
                rviz_config,
                generated_assets_dir / 'multi_drone_px4.rviz',
                drone_namespaces,
            )
            print(f"RViz config file: {rviz_config}")
            rviz = Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=['-d', rviz_config]
            )
            actions.append(RegisterEventHandler(
                OnProcessStart(target_action=px4_drone_control, on_start=[rviz]),
            ))
    elif use_crazyflies == 'True' and use_sim_time:
        actions.append(
            OpaqueFunction(function=render_crazyflie_models)
        )
        updated_world_file = generated_paths['world']
        package_share_dir = Path(get_package_share_directory(PACKAGE_NAME))
        package_models_dir = package_share_dir / 'models'
        gz_env = os.environ.copy()
        append_env_path(gz_env, 'GZ_SIM_RESOURCE_PATH', package_models_dir)
        append_env_path(gz_env, 'IGN_GAZEBO_RESOURCE_PATH', package_models_dir)

        gz_sim = ExecuteProcess(
            cmd=[
                'ros2', 'launch',
                'ros_gz_sim',
                'gz_sim.launch.py',
                'gz_args:=' + str(updated_world_file) + ' -r',
            ],
            env=gz_env,  # type: ignore
            output='screen'
        )
        actions.append(gz_sim)

        spawn_d = TimerAction(period=2.0, actions=[OpaqueFunction(function=spawn_crazyflies, args=[starts])])
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=gz_sim, on_start=[spawn_d]),
        ))

        waypoint_generator = Node(
            package="rbmapf_gzsim",
            executable="waypoint_generator",
            name="waypoint_generator",
            output="screen",
            arguments=[
                '--visual', str(habitat),
                '--config_file', config_file,
                '--num_agents', str(num_drones),
                '--problem_set_file', problem_set_file,
                '--constrained_ckpt_file', constrained_ckpt_file,
                '--unconstrained_ckpt_file', unconstrained_ckpt_file,
            ],
            parameters=[{
                'interface': 'crazyflie',
                'use_sim_time': True,
                'risk_bound_percent': ParameterValue(
                    LaunchConfiguration('risk_bound_percent'),
                    value_type=float,
                ),
                'problem_index': ParameterValue(
                    LaunchConfiguration('problem_index'),
                    value_type=int,
                ),
            }],
        )
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=gz_sim, on_start=[waypoint_generator]),
        ))

        drone_ids = ','.join(str(i + 1) for i in range(num_drones))
        drone_namespaces = [f'/crazyflie_{i + 1}' for i in range(num_drones)]
        drone_namespaces_csv = ','.join(drone_namespaces)

        if habitat:
            habitat_sensor_node = Node(
                package="rbmapf_gzsim",
                executable="habitat_sensor_node",
                name="habitat_sensor_node",
                output="screen",
                parameters=[{
                    'use_sim_time': True,
                    'config_file': config_file,
                    'drone_namespaces': drone_namespaces_csv,
                }],
            )
            actions.append(RegisterEventHandler(
                OnProcessStart(target_action=waypoint_generator, on_start=[habitat_sensor_node]),
            ))

        pkg_project_bringup = get_package_share_directory(PACKAGE_NAME)
        test_bridge_yaml = os.path.join(pkg_project_bringup, 'config', 'gzsim_bridge.yaml')
        with open(test_bridge_yaml, 'r') as f:
            bridge_config_template = yaml.safe_load(f)

        clock_bridge_entries = [
            entry for entry in bridge_config_template
            if entry.get('ros_topic_name') == '/clock'
        ]
        model_bridge_template = [
            entry for entry in bridge_config_template
            if entry.get('ros_topic_name') != '/clock'
        ]

        clock_bridge_config_path = generated_assets_dir / 'bridge_config_clock.yaml'
        clock_bridge_config_path.write_text(yaml.safe_dump(clock_bridge_entries, sort_keys=False))
        clock_bridge = Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='ros_gz_bridge_clock',
            parameters=[{
                'config_file': str(clock_bridge_config_path),
                'use_sim_time': True,
            }],
            output='screen'
        )
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=gz_sim, on_start=[clock_bridge]),
        ))

        bridge = None
        for idx in range(1, num_drones + 1):
            bridge_config_path = generated_assets_dir / f'bridge_config_crazyflie_{idx}.yaml'
            model_bridge_entries = []
            for entry in model_bridge_template:
                rendered_entry = {}
                for key, value in entry.items():
                    if isinstance(value, str):
                        rendered_entry[key] = value.replace('{model_name}', f'crazyflie_{idx}')
                    else:
                        rendered_entry[key] = value
                model_bridge_entries.append(rendered_entry)
            bridge_config_path.write_text(yaml.safe_dump(model_bridge_entries, sort_keys=False))
            bridge = Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                name=f'ros_gz_bridge_crazyflie_{idx}',
                parameters=[{
                    'config_file': str(bridge_config_path),
                    'use_sim_time': True,
                }],
                output='screen'
            )
            files = [config_file, constrained_ckpt_file, str(walls_file)]
            if habitat:
                files.append(str(bounds_file))
            cf_drone_control = Node(
                package="rbmapf_gzsim",
                executable="crazyflie_drone_control",
                name=f"crazyflie_drone_control_{idx}",
                output="screen",
                parameters=[{
                    'files': files,
                    'drone_id': idx,
                    'use_sim_time': True,
                    'visual': str(habitat),
                    'num_drones': num_drones,
                    'drone_ns': f'/crazyflie_{idx}',
                }],
            )

            timed_actions = TimerAction(period=2.0, actions=[cf_drone_control, bridge])
            actions.append(RegisterEventHandler(
                OnProcessStart(target_action=waypoint_generator, on_start=[timed_actions]),
            ))

        multi_tf = Node(
            package="rbmapf_gzsim",
            executable="multi_tf_broadcaster",
            name="multi_tf_broadcaster",
            output="screen",
            parameters=[{'namespaces': drone_namespaces_csv, 'use_sim_time': True}],
        )
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=gz_sim, on_start=[multi_tf]),
        ))

        rviz_config = os.path.join(
            get_package_share_directory('rbmapf_gzsim'),
            'launch',
            'multi_drone.rviz'
        )
        rviz_config = generate_rviz_config(
            rviz_config,
            generated_assets_dir / 'multi_drone_crazyflie_sim.rviz',
            drone_namespaces,
        )
        rviz = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
            parameters=[{
                'use_sim_time': True,
            }],
        )
        delayed_rviz = TimerAction(period=1.0, actions=[rviz])
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=clock_bridge, on_start=[delayed_rviz]),
        ))
        wall_rviz_node = Node(
                package="rbmapf_gzsim",
                executable="wall_rviz_viz",
                name="wall_rviz_viz",
                output="screen",
                arguments=[
                    '--num_agents', str(LaunchConfiguration('num_drones').perform(context)),
                    '--visual', str(habitat),
                    '--sdf_path', str(world_file),
                    '--config_file', config_file,
                    '--problem_set_file', problem_set_file,
                    '--constrained_ckpt_file', constrained_ckpt_file,
                    '--unconstrained_ckpt_file', unconstrained_ckpt_file,
                ],
                parameters=[{'use_sim_time': True}],
            )
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=rviz, on_start=[wall_rviz_node]),
        ))
    else:
        # Using crazyflie hardware

        crazyflie_yaml_path = os.path.join(
            str(generated_assets_dir),
            'crazyflie_mapf.yaml'
        )
        generate_yaml = ExecuteProcess(
            cmd=[
                'ros2', 'run', 'rbmapf_gzsim', 'generate_crazyflie_yaml',
                '--output_path', crazyflie_yaml_path,
                '--num_drones', str(num_drones),
                '--starts_file', str(starts_file),
            ],
            output='screen'
        )
        actions.append(generate_yaml)

        rviz_config = os.path.join(
            get_package_share_directory('rbmapf_gzsim'),
            'launch',
            'multi_drone.rviz'
        )
        hardware_namespaces = [f'/cf{i}' for i in range(1, num_drones + 1)]
        rviz_config = generate_rviz_config(
            rviz_config,
            generated_assets_dir / 'multi_drone_crazyswarm.rviz',
            hardware_namespaces,
        )

        crazyflie_server = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [os.path.join(
                    get_package_share_directory('crazyflie'),
                    'launch',
                    'launch.py'
                )]
            ),
            launch_arguments={
                'crazyflies_yaml_file': crazyflie_yaml_path,
                'rviz_config_file': rviz_config,
                'debug': 'True',
                'rviz': 'False',
                'gui': 'False',
                'teleop': 'False',
                }.items()
        )
        delayed_spawn = TimerAction(period=5.0, actions=[crazyflie_server])
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=generate_yaml, on_start=[delayed_spawn]),
        ))

        rviz = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
            parameters=[{
                'use_sim_time': False,
            }],
        )
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=generate_yaml, on_start=[rviz]),
        ))

        wall_rviz_node = Node(
            package="rbmapf_gzsim",
            executable="wall_rviz_viz",
            name="wall_rviz_viz",
            output="screen",
            arguments=[
                    '--num_agents', str(LaunchConfiguration('num_drones').perform(context)),
                    '--visual', str(habitat),
                    '--sdf_path', str(world_file),
                    '--config_file', config_file,
                    '--problem_set_file', problem_set_file,
                    '--constrained_ckpt_file', constrained_ckpt_file,
                '--unconstrained_ckpt_file', unconstrained_ckpt_file,
            ],
            # parameters=[{'use_sim_time': False}],
        )
        delayed_rviz = TimerAction(period=2.0, actions=[wall_rviz_node])
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=rviz, on_start=[delayed_rviz]),
        ))

        waypoint_generator = Node(
            package="rbmapf_gzsim",
            executable="waypoint_generator",
            name="waypoint_generator",
            output="screen",
            arguments=[
                '--visual', str(habitat),
                '--config_file', config_file,
                '--num_agents', str(num_drones),
                '--use_hardware', str(hardware_demo),
                '--problem_set_file', problem_set_file,
                '--constrained_ckpt_file', constrained_ckpt_file,
                '--unconstrained_ckpt_file', unconstrained_ckpt_file,
            ],
            parameters=[{
                'interface': 'cf',
                'risk_bound_percent': ParameterValue(
                    LaunchConfiguration('risk_bound_percent'),
                    value_type=float,
                ),
                'problem_index': ParameterValue(
                    LaunchConfiguration('problem_index'),
                    value_type=int,
                ),
            }],
        )
        actions.append(RegisterEventHandler(
            OnProcessStart(target_action=rviz, on_start=[waypoint_generator]),
        ))

        # swarm = Crazyswarm()
        for idx in range(1, num_drones + 1):
            cf_drone_control = Node(
                package="rbmapf_gzsim",
                executable="crazyswarm_drone_control",
                name=f"crazyswarm_drone_control_{idx}",
                output="screen",
                parameters=[{
                    'drone_id': idx,
                    'drone_ns': f'/cf{idx}',
                    'num_drones': num_drones,
                    'files': [config_file, constrained_ckpt_file, str(walls_file)],
                }],
                # arguments=[swarm],
            )

            timed_actions = TimerAction(period=2.0, actions=[cf_drone_control])
            actions.append(RegisterEventHandler(
                OnProcessStart(target_action=waypoint_generator, on_start=[timed_actions]),
            ))

    if use_sim_time and gz_version not in ['classic', 'harmonic']:
        raise RuntimeError('Incorrect gz_version provided. Options include classic or harmonic')

    return actions


def launch_setup(context, *args, **kwargs):
    world_file = str(resolve_package_asset(LaunchConfiguration('world_file').perform(context)))
    generated_assets_dir = str(Path(LaunchConfiguration('generated_assets_dir').perform(context)).expanduser())
    config_file = LaunchConfiguration('config_file').perform(context)
    habitat = LaunchConfiguration('habitat').perform(context) == 'True'
    problem_set_file = LaunchConfiguration('problem_set_file').perform(context)
    hardware_demo = LaunchConfiguration('use_hardware').perform(context) == 'True'
    constrained_ckpt_file = LaunchConfiguration('constrained_ckpt_file').perform(context)
    unconstrained_ckpt_file = LaunchConfiguration('unconstrained_ckpt_file').perform(context)

    actions = []

    use_sim_time = not hardware_demo

    wall_spawner_node = Node(
            package="rbmapf_gzsim",
            executable="wall_spawner",
            name="wall_spawner",
            output="screen",
            arguments=[
                '--num_agents', str(LaunchConfiguration('num_drones').perform(context)),
                '--visual', str(habitat),
                '--sdf_path', world_file,
                '--generated_assets_dir', generated_assets_dir,
                '--config_file', config_file,
                '--use_hardware', str(hardware_demo),
                '--problem_set_file', problem_set_file,
                '--constrained_ckpt_file', constrained_ckpt_file,
                '--unconstrained_ckpt_file', unconstrained_ckpt_file,
            ],
            parameters=[{
                'use_sim_time': use_sim_time,
                'risk_bound_percent': ParameterValue(
                    LaunchConfiguration('risk_bound_percent'),
                    value_type=float,
                ),
                'problem_index': ParameterValue(
                    LaunchConfiguration('problem_index'),
                    value_type=int,
                ),
            }],
        )
    actions.append(wall_spawner_node)

    actions.append(
      RegisterEventHandler(
        OnProcessExit(
          target_action=wall_spawner_node,
          on_exit=[OpaqueFunction(function=spawn_processes)]
        )
      )
    )

    return actions


def generate_launch_description():
    ld = LaunchDescription()

    params_file = get_package_share_directory('rbmapf_gzsim') + '/config/parameters.yaml'
    specs = yaml.safe_load(Path(params_file).read_text())

    for name, meta in specs.items():
        default = meta.get("default")
        description = meta.get("description", "")
        if default is None:
            ld.add_action(DeclareLaunchArgument(name, description=description))
        else:
            ld.add_action(DeclareLaunchArgument(
                name,
                default_value=str(default),
                description=description
            ))

    ld.add_action(OpaqueFunction(function=launch_setup))
    return ld
