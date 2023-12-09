package com.springboot.springbootlogindemo.service;

import com.springboot.springbootlogindemo.domain.UAVImage;
import com.springboot.springbootlogindemo.repository.UAVImageRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Optional;

@Service
public class UAVImageService {
    private final UAVImageRepository uavImageRepository;

    @Value("${image.upload.path}") // 从配置文件中读取图片上传路径
    private String imageUploadPath;

    @Autowired
    public UAVImageService(UAVImageRepository uavImageRepository) {
        this.uavImageRepository = uavImageRepository;
    }

    // 保存图片到服务器和数据库
    public UAVImage saveImage(MultipartFile file, Integer name, Integer class1, Integer class2, Float weight) throws IOException {
        // 生成新的文件名
        String originalFilename = file.getOriginalFilename();
        String fileExtension = originalFilename.substring(originalFilename.lastIndexOf("."));
        String newFileName = System.currentTimeMillis() + fileExtension;

        // 保存图片到服务器指定路径
        File targetFile = new File(imageUploadPath, newFileName);
        file.transferTo(targetFile);

        // 保存图片信息到数据库
        UAVImage uavImage = new UAVImage();
        uavImage.setName(name);
        uavImage.setClass1(class1);
        uavImage.setClass2(class2);
        uavImage.setWeight(weight);
        // 构建完整的文件路径，例如：C:\\Users\\10987\\Desktop\\tong\\文件名.jpg
        String filePath = imageUploadPath + File.separator + newFileName;
        uavImage.setPath(filePath);


        return uavImageRepository.save(uavImage);
    }

    // 查找所有图片
    public List<UAVImage> findAllImages() {
        return uavImageRepository.findAll();
    }

    // 根据ID删除图片
    public void deleteImageById(Long id) {
        // 先从数据库获取图片信息
        UAVImage uavImage = uavImageRepository.findById(id).orElse(null);
        if (uavImage != null) {
            // 删除服务器上的图片文件
            File targetFile = new File(imageUploadPath, uavImage.getPath());
            if (targetFile.exists()) {
                targetFile.delete();
            }
            // 删除数据库中的图片信息
            uavImageRepository.deleteById(id);
        }
    }


    public UAVImage findImageById(Long id) {
        return uavImageRepository.findById(id).orElse(null);
    }


    public String getImagePathById(Long id) {
        Optional<UAVImage> imageOptional = uavImageRepository.findById(id);
        return imageOptional.map(UAVImage::getPath).orElse(null);
    }

}
