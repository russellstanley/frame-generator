#include <dv-sdk/module.hpp>
#include <opencv2/imgproc.hpp>

class FrameGenerator : public dv::ModuleBase
{
private:
	cv::Size inputSize;
	cv::Vec3b onColor;
	cv::Vec3b offColor;

public:
	static void initInputs(dv::InputDefinitionList &in)
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
		config.add("red", dv::ConfigOption::intOption("Value of the red color component", 255, 0, 255));
		config.add("green", dv::ConfigOption::intOption("Value of the green color component", 255, 0, 255));
		config.add("blue", dv::ConfigOption::intOption("Value of the blue color component", 255, 0, 255));

		config.setPriorityOptions({"red", "green", "blue"});
	}

	FrameGenerator()
	{
		outputs.getFrameOutput("frames").setup(inputs.getEventInput("events"));
		inputSize = inputs.getEventInput("events").size();
	}

	void configUpdate() override
	{
		onColor = cv::Vec3b(static_cast<uint8_t>(config.getInt("blue")), static_cast<uint8_t>(config.getInt("green")),
							static_cast<uint8_t>(config.getInt("red")));

		offColor = cv::Vec3b(255 - static_cast<uint8_t>(config.getInt("blue")), 255 - static_cast<uint8_t>(config.getInt("green")),
							 255 - static_cast<uint8_t>(config.getInt("red")));
	}

	void run() override
	{
		auto events = inputs.getEventInput("events").events();
		cv::Mat outFrame = cv::Mat::zeros(inputSize, CV_8UC3);

		for (const auto &event : events)
		{
			if (event.polarity())
			{
				outFrame.at<cv::Vec3b>(event.y(), event.x()) = onColor;
			}
			else
			{
				outFrame.at<cv::Vec3b>(event.y(), event.x()) = offColor;
			}
		}

		outputs.getFrameOutput("frames") << outFrame << dv::commit;
	};
};

registerModuleClass(FrameGenerator)