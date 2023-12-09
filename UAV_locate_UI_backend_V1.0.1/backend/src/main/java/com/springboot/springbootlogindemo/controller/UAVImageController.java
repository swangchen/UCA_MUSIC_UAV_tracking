package com.springboot.springbootlogindemo.controller;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import com.springboot.springbootlogindemo.domain.UAVImage;
import com.springboot.springbootlogindemo.service.UAVImageService;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

@RestController
@RequestMapping("/api/images")
public class UAVImageController {
    private final UAVImageService uavImageService;

    @Autowired
    public UAVImageController(UAVImageService uavImageService) {
        this.uavImageService = uavImageService;
    }

    // 上传图片
    @CrossOrigin(origins = "*")
    @PostMapping("/upload")
    public ResponseEntity<String> uploadImage(@RequestParam("file") MultipartFile file, @RequestParam("name") Integer name, @RequestParam("class1") Integer class1, @RequestParam("class2") Integer class2, @RequestParam("weight") Float weight) {
        try {
            UAVImage uavImage = uavImageService.saveImage(file, name, class1, class2, weight);
            if (uavImage != null) {
                return new ResponseEntity<>("文件上传成功", HttpStatus.OK);
            } else {
                return new ResponseEntity<>("文件上传失败", HttpStatus.INTERNAL_SERVER_ERROR);
            }
        } catch (IOException e) {
            e.printStackTrace();
            return new ResponseEntity<>("文件上传失败", HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    // 查找所有图片
    @GetMapping("/all")
    public ResponseEntity<List<UAVImage>> findAllImages() {
        List<UAVImage> images = uavImageService.findAllImages();
        return new ResponseEntity<>(images, HttpStatus.OK);
    }

    // 根据ID删除图片
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteImage(@PathVariable Long id) {
        uavImageService.deleteImageById(id);
        return new ResponseEntity<>(HttpStatus.NO_CONTENT);
    }

    // 根据ID查找图片
    @CrossOrigin(origins = "*")
    @GetMapping("/find/{id}")
    public ResponseEntity<UAVImage> findImageById(@PathVariable Long id) {
        UAVImage uavImage = uavImageService.findImageById(id);
        if (uavImage != null) {
            return new ResponseEntity<>(uavImage, HttpStatus.OK);
        } else {
            return new ResponseEntity<>(HttpStatus.NOT_FOUND);
        }
    }

    @CrossOrigin(origins = "*")
    @GetMapping("/showimg/{id}")
    public ResponseEntity<String> getImagePathById(@PathVariable Long id) {
        String imagePath = uavImageService.getImagePathById(id);
        if (imagePath != null) {
            return new ResponseEntity<>(imagePath, HttpStatus.OK);
        } else {
            return new ResponseEntity<>("Image not found", HttpStatus.NOT_FOUND);
        }
    }
}
