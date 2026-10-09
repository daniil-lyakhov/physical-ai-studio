import { useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    Divider,
    Flex,
    Heading,
    Item,
    Picker,
    Radio,
    RadioGroup,
    Text,
} from '@geti-ui/ui';

type Delegate = 'portable' | 'xnnpack' | 'openvino';

export const ExecuTorchExportDialog = ({ close }: { close: () => void }) => {
    const [delegate, setDelegate] = useState<Delegate>('portable');
    const [device, setDevice] = useState('CPU');
    const [previewed, setPreviewed] = useState(false);
    const selection = delegate === 'openvino' ? `OpenVINO on ${device}` : delegate;

    return (
        <Dialog>
            <Heading>Export to ExecuTorch</Heading>
            <Divider />
            <Content>
                <Flex direction='column' gap='size-200'>
                    <Text>Choose how the model will run at the edge.</Text>
                    <RadioGroup
                        aria-label='Delegate'
                        value={delegate}
                        onChange={(value) => {
                            setDelegate(value as Delegate);
                            setPreviewed(false);
                        }}
                    >
                        <Flex direction='column' gap='size-100'>
                            <Radio value='portable'>Portable — compatible CPU execution</Radio>
                            <Radio value='xnnpack'>XNNPACK — optimized CPU execution</Radio>
                            <Radio value='openvino'>OpenVINO — Intel acceleration</Radio>
                        </Flex>
                    </RadioGroup>
                    {delegate === 'openvino' && (
                        <Picker
                            label='Target device'
                            selectedKey={device}
                            onSelectionChange={(key) => {
                                setDevice(String(key));
                                setPreviewed(false);
                            }}
                        >
                            <Item key='CPU'>CPU</Item>
                            <Item key='GPU'>GPU</Item>
                            <Item key='NPU'>NPU</Item>
                        </Picker>
                    )}
                    <Text>UI preview only. Export is not connected; no artifact will be created.</Text>
                    {previewed && (
                        <div role='status'>
                            <Text>Preview: {selection} selected. No model was exported.</Text>
                        </div>
                    )}
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close}>
                    Close
                </Button>
                <Button variant='accent' onPress={() => setPreviewed(true)}>
                    Export (preview)
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
