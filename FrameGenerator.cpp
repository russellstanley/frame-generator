#include <dv-sdk/module.hpp>
#include <dv-sdk/processing/frame.hpp>
#include <opencv2/imgproc.hpp>

class FrameGenerator : public dv::ModuleBase
{
private:
	cv::Size inputSize;
	dv::TimeSurface timeSurface;
	cv::Mat outFrame;

public:
	static void
	initInputs(dv::InputDefinitionList &in)
	{
		in.addEventInput("events");
	}

	static void initOutputs(dv::OutputDefinitionList &out)
	{
		out.addFrameOutput("frames");
	}

	static const char *initDescription()
	{
		return ("This module renders all events to frames");
	}

	static void initConfigOptions(dv::RuntimeConfig &config)
	{
		config.add("R", dv::ConfigOption::intOption("R", 32, 0, 32));
		config.add("Lookback", dv::ConfigOption::intOption("Time to look back", 100, 100, 100000));
		config.setPriorityOptions({"R"});
	}

	FrameGenerator()
	{
		outputs.getFrameOutput("frames").setup(inputs.getEventInput("events"));
		inputSize = inputs.getEventInput("events").size();
		timeSurface = dv::TimeSurface(inputSize);
	}

	void configUpdate() override
	{
	}

	void run() override
	{
		cv::Mat outFrame = cv::Mat::zeros(inputSize, CV_8UC3);

		timeSurface.accept(inputs.getEventInput("events").events());

		outFrame = timeSurface.getOCVMatScaled(config.getInt("Lookback"));

		outputs.getFrameOutput("frames") << outFrame << dv::commit;
	};
};

registerModuleClass(FrameGenerator)