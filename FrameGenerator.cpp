#include <dv-sdk/module.hpp>

class FrameGenerator : public dv::ModuleBase
{
private:
	long numberOfEvents = 0;

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
		return ("This module will generate event frames");
	}

	static void initConfigOptions(dv::RuntimeConfig &config)
	{
		config.add("printInterval",
				   dv::ConfigOption::intOption(
					   "Interval in number of events between consecutive printing of the event number.", 10000, 1, INT_MAX));
	}

	void run() override
	{
		auto printInterval = config.getInt("printInterval");

		auto events = inputs.getEventInput("events").events();

		for (const auto &event : events)
		{
			if (event.polarity())
			{
				numberOfEvents += 1;

				if (numberOfEvents % printInterval == 0)
				{
					log.info << "Processed " << numberOfEvents << " positive events" << dv::logEnd;
				}
			}
		}
	}
};

registerModuleClass(FrameGenerator)
