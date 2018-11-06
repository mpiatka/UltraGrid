#include "combobox_ui.hpp"

ComboBoxUi::ComboBoxUi(QComboBox *box,
        Settings *settings,
        const std::string &opt,
        std::function<std::vector<SettingItem>()> itemBuilder) :
    WidgetUi(settings, opt),
    box(box),
    itemBuilder(itemBuilder)
{
    refresh();
    updateUiState();
    registerCallback();
    connectSignals();
}

void ComboBoxUi::connectSignals(){
	connect(box, QOverload<int>::of(&QComboBox::activated),
			this, &ComboBoxUi::itemSelected);
}

void ComboBoxUi::addItem(const SettingItem &item){
		box->addItem(QString::fromStdString(item.name),
				QVariant::fromValue(item));

        for(const auto &option : item.opts){
            registerCallback(option.opt);
        }
}

void ComboBoxUi::refresh(){
    std::vector<SettingItem> newItems = itemBuilder();

    box->clear();
    for(const auto &item : newItems){
        addItem(item);
    }

    updateUiState();
}

static bool conditionsSatisfied(
        const std::vector<std::vector<SettingValue>> &conds,
        Settings *settings)
{
    for(const auto &condGroup : conds){
        bool orRes = false;
        for(const auto &itemCond : condGroup){
            if(itemCond.val == settings->getOption(itemCond.opt).getValue()){
                orRes = true;
                break;
            }
        }
        if(!orRes){
            return false;
        }
    }

    return true;
}

void ComboBoxUi::selectOption(){
    int i;
    bool found = false;
    for(i = 0; i < box->count(); i++){
        const SettingItem &item = box->itemData(i).value<SettingItem>();

        if(!conditionsSatisfied(item.conditions, settings)){
            continue;
        }

        found = true;
        for(const auto &itemOpt : item.opts){
            if(itemOpt.val != settings->getOption(itemOpt.opt).getValue()){
                found = false;
                break;
            }
        }
        
        if(found)
            break;
    }

    if(found){
        box->setCurrentIndex(i);
    }
}

void ComboBoxUi::optChangeCallback(Option &opt, bool suboption){
    selectOption();
}

void ComboBoxUi::updateUiState(){
    selectOption();
}

void ComboBoxUi::itemSelected(int index){
	const SettingItem &item = box->itemData(index).value<SettingItem>();

	for(const auto &option : item.opts){
		settings->getOption(option.opt).setValue(option.val);
	}

    emit changed();
}

