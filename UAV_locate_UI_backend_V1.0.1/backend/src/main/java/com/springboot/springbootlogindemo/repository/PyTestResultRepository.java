package com.springboot.springbootlogindemo.repository;

import com.springboot.springbootlogindemo.domain.PyTestResult;
import org.springframework.data.jpa.repository.JpaRepository;

public interface PyTestResultRepository extends JpaRepository<PyTestResult, Long> {
    // 这里可以定义自定义的查询方法，例如根据某个条件检索数据
}
