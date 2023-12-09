package com.springboot.springbootlogindemo.repository;


import com.springboot.springbootlogindemo.domain.UAVImage;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Map;

@Repository
public interface UAVImageRepository extends JpaRepository<UAVImage, Long> {

    // 查找所有图片
    List<UAVImage> findAll();


    // 保存新的图片
    UAVImage save(UAVImage uavimage);

    // 根据ID删除特定图片
    void deleteById(Long id);

}
