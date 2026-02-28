import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime


def create_save_dir():
    """创建保存图像的目录（按日期分类）"""
    root_dir = "realsense_images"
    date_dir = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(root_dir, date_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def save_images(color_img, depth_img, depth_colormap, save_dir):
    """保存图像：彩色图、深度可视化图、深度原始数据"""
    timestamp = datetime.now().strftime("%H-%M-%S-%f")[:-3]
    # 保存彩色图
    color_path = os.path.join(save_dir, f"color_{timestamp}.jpg")
    cv2.imwrite(color_path, color_img)
    # 保存深度可视化图
    depth_colormap_path = os.path.join(save_dir, f"depth_colormap_{timestamp}.jpg")
    cv2.imwrite(depth_colormap_path, depth_colormap)
    # 保存深度原始数据
    depth_raw_path = os.path.join(save_dir, f"depth_raw_{timestamp}.png")
    cv2.imwrite(depth_raw_path, depth_img)
    print(
        f"图像保存完成：\n  彩色图：{color_path}\n  深度可视化图：{depth_colormap_path}\n  深度原始数据：{depth_raw_path}")


def get_depth_range(depth_img):
    """获取当前深度帧的有效范围（过滤0值，模仿Viewer的逻辑）"""
    valid_depth = depth_img[depth_img > 0]
    if len(valid_depth) == 0:
        return 0, 6000
    min_depth = np.min(valid_depth)
    max_depth = np.max(valid_depth)
    min_depth = max(min_depth, 300)
    max_depth = min(max_depth, 3000)
    return min_depth, max_depth


def main():
    save_dir = create_save_dir()
    print(f"图像将保存至：{save_dir}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline_started = False  # 标记pipeline是否启动成功
    try:
        # 启动相机
        profile = pipeline.start(config)
        pipeline_started = True

        # 配置夜间拍摄参数
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.emitter_enabled, 1)
        depth_sensor.set_option(rs.option.exposure, 2000)

        # 对齐深度和彩色帧
        align = rs.align(rs.stream.color)

        print("相机启动成功！")
        print("操作说明：")
        print("  - 按 's' 键保存当前帧（彩色图+深度图）")
        print("  - 按 'q' 键退出程序")

        while True:
            # 获取并对齐帧
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # 转换为numpy数组
            depth_img = np.asanyarray(depth_frame.get_data())
            color_img = np.asanyarray(color_frame.get_data())

            # 动态缩放深度值（匹配Viewer颜色）
            min_depth, max_depth = get_depth_range(depth_img)
            depth_normalized = (depth_img - min_depth) / (max_depth - min_depth)
            depth_normalized = np.clip(depth_normalized, 0, 1)
            depth_scaled = (depth_normalized * 255).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)

            # 拼接显示图像
            combined_img = np.hstack((color_img, depth_colormap))
            cv2.putText(combined_img, "Press 's' to save | Press 'q' to quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined_img, f"Depth Range: {min_depth / 1000:.1f}m ~ {max_depth / 1000:.1f}m",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.imshow("Intel RealSense D435F (Save Image)", combined_img)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("程序退出中...")
                break
            elif key == ord('s'):
                save_images(color_img, depth_img, depth_colormap, save_dir)

    except Exception as e:
        print(f"程序运行出错：{type(e).__name__} - {e}")
        print("\n排查建议：")
        print("1. 确认相机已连接到USB 3.0端口（蓝色接口）")
        print("2. 确认RealSense SDK已安装且版本匹配")
        print("3. 确认pyrealsense2库已正确安装（pip install pyrealsense2）")

    finally:
        if pipeline_started:
            pipeline.stop()
        cv2.destroyAllWindows()
        print("\n程序结束，所有资源已释放")


# 主函数调用
if __name__ == "__main__":
    main()