from src.modules.dataloader import DataLoader

def test_detaloader():
    dataloader = DataLoader('./test/fixtures/position.json')
    assert dataloader.annotation_file_path == './test/fixtures/position.json'
    assert dataloader.image_list == ["./test/fixtures/images/01.jpg","./test/fixtures/images/02.jpeg","./test/fixtures/images/03.png"]
    assert dataloader.label_list == [
            [[369,338],[304,94],[339,136],[389,199],[161,216],[211,227],[264,252],[126,332],[184,329],[250,329],[164,431],[212,413],[265,395],[218,508],[252,483],[293,457]],
            [[289,450],[470,271],[415,317],[369,375],[132,217],[177,253],[226,296],[252,420],[190,375],[223,339],[244,449],[181,420],[178,394],[225,483],[171,463],[160,432]],
            [[229,308],[58,159],[105,205],[132,264],[358,118],[312,154],[274,207],[467,175],[395,204],[320,248],[464,215],[398,246],[337,277],[460,325],[405,333],[343,341]]
        ]

    dataset = dataloader.load_data()
    assert dataset.__len__() == 3

    train, valid, test = dataloader.split(ratio = (1, 1, 1))
    assert (train.__len__(), valid.__len__(), test.__len__()) == (1, 1, 1)