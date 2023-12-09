package com.springboot.springbootlogindemo.service;
import com.springboot.springbootlogindemo.domain.PyTestResult;
import com.springboot.springbootlogindemo.repository.PyTestResultRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;

@Service
public class PyTestService {
    private final PyTestResultRepository pyTestResultRepository;
    private final Environment environment;

    @Autowired
    public PyTestService(PyTestResultRepository pyTestResultRepository, Environment environment) {
        this.pyTestResultRepository = pyTestResultRepository;
        this.environment = environment;
    }

    // 运行Python脚本并将结果上传到数据库
    public void runPyScriptAndSaveToDatabase() {
        try {
            // 获取Python解释器路径
            String pythonInterpreter = environment.getProperty("python");

            // 构建Python脚本运行命令
            ProcessBuilder processBuilder = new ProcessBuilder(pythonInterpreter, "python-script\\test.py");
            processBuilder.redirectErrorStream(true);

            // 启动Python脚本进程
            Process process = processBuilder.start();

            // 读取Python脚本的输出
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            StringBuilder output = new StringBuilder();
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }

            // 等待Python脚本执行完成
            int exitCode = process.waitFor();

            if (exitCode == 0) {
                // Python脚本成功执行
                String scriptOutput = output.toString();

                // 将结果保存到PyTestResult实体中并使用Repository插入到数据库
                PyTestResult result = new PyTestResult();
                result.setResult(Double.parseDouble(scriptOutput)); // 假设脚本输出是一个浮点数
                pyTestResultRepository.save(result);
            } else {
                // Python脚本执行失败
                System.err.println("Python脚本执行失败");
            }
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    // 从数据库中获取计算结果
    public List<PyTestResult> getPyTestResults() {
        return pyTestResultRepository.findAll();
    }
}
