package com.springboot.springbootlogindemo.controller;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.*;

@RestController
@RequestMapping("/api")
public class PythonController {

    @Value("${python}")
    private String pythonPath;


    @GetMapping("/runPythonScript")
    public String runPythonScript() {
        try {
            // 指定Python脚本的路径
            String pythonScriptPath = "python-script/test.py";

            // 构建Python命令
            String[] command = {pythonPath, pythonScriptPath}; // 使用配置的Python路径

            // 启动Python脚本并获取输出
            Process process = new ProcessBuilder(command).start();
            InputStream inputStream = process.getInputStream();
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            StringBuilder output = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }

            // 等待进程完成
            int exitCode = process.waitFor();

            if (exitCode == 0) {
                return "Python脚本执行成功，输出为:\n" + output.toString();
            } else {
                return "Python脚本执行失败，输出为:\n" + output.toString();
            }
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
            return "Python脚本执行出错: " + e.getMessage();
        }
    }
    @GetMapping("/train")
        public String pyfileUpload()   {
        String result = "";
        try {
            //这个方法是类似隐形开启了命令执行器，输入指令执行python脚本
            Process process = Runtime.getRuntime()
                    .exec("python=C:\\Users\\10987\\anaconda3\\envs\\pytorch\\python.exe C:\\Users\\10987\\Desktop\\Engineering\\杜宇\\train.py）");
            //这种方式获取返回值的方式是需要用python打印输出，然后java去获取命令行的输出，在java返回
            InputStreamReader ir = new InputStreamReader(process.getInputStream());
            LineNumberReader input = new LineNumberReader(ir);
            result = input.readLine();//中文的话这里可能会有乱码，可以尝试转一下不过问题不大
//            result1 = new String(result.getBytes("iso8859-1"),"utf-8");
            input.close();
            ir.close();
            int re = process.waitFor();
            System.out.println(result);
        } catch (IOException | InterruptedException e) {
            System.out.println("调用python脚本并读取结果时出错：" + e.getMessage());
        }
        return result;
    }

    @GetMapping("/predict")
    public String test() {
        try {
            // 指定Python脚本的路径
            String pythonScriptPath = "python-script/predict.py";

            // 构建Python命令
            String[] command = {pythonPath, pythonScriptPath}; // 使用配置的Python路径

            // 启动Python脚本并获取输出
            Process process = new ProcessBuilder(command).start();
            InputStream inputStream = process.getInputStream();
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            StringBuilder output = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }

            // 等待进程完成
            int exitCode = process.waitFor();

            if (exitCode == 0) {
                return "Python脚本执行成功，输出为:\n" + output.toString();
            } else {
                return "Python脚本执行失败，输出为:\n" + output.toString();
            }
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
            return "Python脚本执行出错: " + e.getMessage();
        }
    }
    @GetMapping("/evaluate")
    public String evaluate() {
        try {
            // 指定Python脚本的路径
            String pythonScriptPath = "python-script/evaluate.py";

            // 构建Python命令
            String[] command = {pythonPath, pythonScriptPath}; // 使用配置的Python路径

            // 启动Python脚本并获取输出
            Process process = new ProcessBuilder(command).start();
            InputStream inputStream = process.getInputStream();
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            StringBuilder output = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }

            // 等待进程完成
            int exitCode = process.waitFor();

            if (exitCode == 0) {
                return "Python脚本执行成功，输出为:\n" + output.toString();
            } else {
                return "Python脚本执行失败，输出为:\n" + output.toString();
            }
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
            return "Python脚本执行出错: " + e.getMessage();
        }
    }



}
