% 设置文件夹路径和保存路径
folder_path = ''; % 图片所在的文件夹路径
save_path = ''; % 结果保存的文件夹路径

% 获取文件夹中所有的图片文件名
file_list = dir(fullfile(folder_path, '*.png')); % 根据后缀名获取所有的 PNG 图片

% 遍历文件夹中的所有图片
for i = 1:length(file_list)
    % 读入图片
    file_name = file_list(i).name;
    im = imread(fullfile(folder_path, file_name));

    % 将图像转换为灰度图像
    im_gray = rgb2gray(im);

    % 调用 VerticalProject 函数进行垂直投影，并获取投影后的灰度强度向量
    project_intensity_Vertical = VerticalProject(im_gray);

    % 将灰度强度向量保存在 Excel 表中
    [~, file_name_prefix, ~] = fileparts(file_name);
    filename = [file_name_prefix, '_project_intensity_Vertical.xlsx'];
    sheet = 'Sheet1';
    xlswrite(fullfile(save_path, filename), project_intensity_Vertical, sheet);

    % 使用 findpeaks 函数找到显著度至少为 2 的峰值点，并在图中标识出来
    [peaks, locs, ~, proms] = findpeaks(project_intensity_Vertical, 'MinPeakProminence', 3000, 'Annotate', 'extents');
    
    % 绘制灰度强度曲线图
    fig = figure('Visible', 'off');
    plot(project_intensity_Vertical);
    xlabel('列数');
    ylabel('灰度强度');
    title('超声图像垂直投影');

    hold on;
    idx = proms >= 3000;
    plot(locs(idx), peaks(idx), 'rv');
    text(locs(idx)+2, peaks(idx)+0.02*max(project_intensity_Vertical), ...
        compose('%s (%.2f)', num2str(locs(idx)), proms(idx)), 'Color', 'red');
    hold off;
    legend('灰度强度', '峰值点');

    % 保存结果图片
    save_filename = [file_name_prefix, '_result.png'];
    saveas(fig, fullfile(save_path, save_filename));
end